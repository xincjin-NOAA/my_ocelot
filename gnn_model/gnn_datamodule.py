import os
import importlib
import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
import zarr
from zarr.storage import LRUStoreCache
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

from nnja_adapter import build_zlike_from_df
from process_timeseries import extract_features, organize_bins_times
from create_mesh_graph_global import obs_mesh_conn


def _t32(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


# -------------------------
# Dataset per bin
# -------------------------
class BinDataset(Dataset):
    def __init__(
        self,
        bin_names,
        data_summary,
        zarr_store,
        create_graph_fn,
        observation_config,
        feature_stats=None,
        tag="TRAIN",
    ):
        self.bin_names = list(bin_names) if bin_names is not None else []
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.tag = tag

    def __len__(self):
        return len(self.bin_names)

    def __getitem__(self, idx):
        bin_name = self.bin_names[idx]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"[Rank {rank}] [{self.tag}] fetching {bin_name} ... ds_id={id(self)} sum_id={id(self.data_summary)}")
        try:
            out = extract_features(
                self.z,
                self.data_summary,
                bin_name,
                self.observation_config,
                feature_stats=self.feature_stats,
            )
            bin_data = out[bin_name]
            graph_data = self.create_graph_fn(bin_data)
            graph_data.bin_name = bin_name
            return graph_data
        except Exception as e:
            print(f"[Rank {rank}] [{self.tag}] ERROR processing {bin_name}: {e}")
            raise


# -------------------------
# DataModule
# -------------------------
class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        start_date,
        end_date,
        observation_config,
        mesh_structure,
        batch_size=1,
        num_neighbors=3,
        feature_stats=None,
        latent_step_hours=None,   # latent rollout support
        window_size="12h",        # binning window
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mesh_structure = mesh_structure
        self.feature_stats = feature_stats

        # Zarr handles (stable across window changes)
        self.z = None

        # Separate train/val summaries + bin name lists
        self.train_data_summary = None
        self.val_data_summary = None
        self.train_bin_names = []
        self.val_bin_names = []

        # Version counters (for debugging staleness)
        self._train_version = 0
        self._val_version = 0

        # If callbacks want separate windows, they will set these:
        # default: both train/val start with start_date..end_date
        self.hparams.train_start = pd.to_datetime(kwargs.get("train_start", start_date))
        self.hparams.train_end   = pd.to_datetime(kwargs.get("train_end", end_date))
        self.hparams.val_start   = pd.to_datetime(kwargs.get("val_start", start_date))
        self.hparams.val_end     = pd.to_datetime(kwargs.get("val_end", end_date))

    # ------------- Setup / Zarr open -------------
    def setup(self, stage=None):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # Open Zarrs once
        if self.z is None:
            self.z = {}
            for obs_type, instruments in self.hparams.observation_config.items():
                self.z[obs_type] = {}
                for inst_name, inst_cfg in instruments.items():
                    src = inst_cfg.get("source", "zarr")

                    if src == "zarr":
                        zarr_dir = inst_cfg.get("zarr_dir")
                        if zarr_dir:
                            zarr_path = zarr_dir
                        else:
                            zname = inst_cfg.get("zarr_name", inst_name)
                            if not zname.endswith(".zarr"):
                                zname += ".zarr"
                            zarr_path = os.path.join(self.hparams.data_path, zname)

                        if not os.path.isdir(zarr_path):
                            raise FileNotFoundError(f"Zarr not found: {zarr_path}")

                        # Use LRU cache; ensure int for max_size
                        store = LRUStoreCache(zarr.DirectoryStore(zarr_path), max_size=int(2e9))
                        self.z[obs_type][inst_name] = zarr.open(store, mode="r")

                        if rank == 0:
                            print(f"[ZARR] {obs_type}/{inst_name} -> {zarr_path}")
                            try:
                                print("       keys:", list(self.z[obs_type][inst_name].keys())[:12])
                            except Exception:
                                pass

                        if obs_type == "conventional" and inst_name == "surface_obs":
                            if not os.path.basename(zarr_path).startswith("raw_surface_obs"):
                                print(f"[WARN] surface_obs expected raw_surface_obs*.zarr but got: {zarr_path}")

                    elif src == "nnja":
                        loader_path = inst_cfg["dataframe_loader"]
                        mod_name, fn_name = loader_path.rsplit(".", 1)
                        load_fn = getattr(importlib.import_module(mod_name), fn_name)

                        need = list(inst_cfg["var_map"].values())
                        need += [
                            inst_cfg.get("lat_col", "LAT"),
                            inst_cfg.get("lon_col", "LON"),
                            inst_cfg.get("time_col", "OBS_TIMESTAMP"),
                        ]

                        df = load_fn(
                            start_date=self.hparams.start_date,  # initial window used for first setup
                            end_date=self.hparams.end_date,
                            columns=need,
                        )
                        self.z[obs_type][inst_name] = build_zlike_from_df(
                            df,
                            var_map=inst_cfg["var_map"],
                            lat_col=inst_cfg.get("lat_col", "LAT"),
                            lon_col=inst_cfg.get("lon_col", "LON"),
                            time_col=inst_cfg.get("time_col", "OBS_TIMESTAMP"),
                        )
                    else:
                        raise ValueError(f"Unknown source '{src}' for {inst_name}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Build TRAIN and VAL summaries for current windows
        print(
            f"[Rank {rank}] [DM.setup stage={stage}] "
            f"train_window={self.hparams.train_start}..{self.hparams.train_end} | "
            f"val_window={self.hparams.val_start}..{self.hparams.val_end}"
        )

        self._rebuild_train_summary()
        self._rebuild_val_summary()

        if stage in (None, "fit"):
            # For now we use the full lists produced by organize_bins_times;
            # callbacks can narrow them by changing windows and triggering reload.
            pass

    # ------------- Summary (re)builders -------------
    def _rebuild_train_summary(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.train_data_summary = organize_bins_times(
            self.z,
            self.hparams.train_start,
            self.hparams.train_end,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
            window_size=self.hparams.window_size,
            latent_step_hours=self.hparams.latent_step_hours,
        )
        self.train_bin_names = sorted(self.train_data_summary.keys(), key=lambda x: int(x.replace("bin", "")))
        print(
            f"[Rank {rank}] [DM.train_summary] v{self._train_version} sum_id={id(self.train_data_summary)} "
            f"bins={len(self.train_bin_names)} first={self.train_bin_names[0] if self.train_bin_names else None}"
        )

    def _rebuild_val_summary(self):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.val_data_summary = organize_bins_times(
            self.z,
            self.hparams.val_start,
            self.hparams.val_end,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
            window_size=self.hparams.window_size,
            latent_step_hours=self.hparams.latent_step_hours,
        )
        self.val_bin_names = sorted(self.val_data_summary.keys(), key=lambda x: int(x.replace("bin", "")))
        print(
            f"[Rank {rank}] [DM.val_summary]   v{self._val_version} sum_id={id(self.val_data_summary)} "
            f"bins={len(self.val_bin_names)} first={self.val_bin_names[0] if self.val_bin_names else None}"
        )

    # ------------- Window setters for callbacks -------------
    def set_train_window(self, start_dt, end_dt):
        self.hparams.train_start = pd.to_datetime(start_dt)
        self.hparams.train_end = pd.to_datetime(end_dt)
        self._train_version += 1
        print(f"[DM.set_train_window] v{self._train_version} -> {self.hparams.train_start} .. {self.hparams.train_end}")
        # Rebuild summary/bin names immediately so the *next* dataloader reload sees fresh objects
        self._rebuild_train_summary()

    def set_val_window(self, start_dt, end_dt):
        self.hparams.val_start = pd.to_datetime(start_dt)
        self.hparams.val_end = pd.to_datetime(end_dt)
        self._val_version += 1
        print(f"[DM.set_val_window]   v{self._val_version} -> {self.hparams.val_start} .. {self.hparams.val_end}")
        self._rebuild_val_summary()

    # ------------- Graph builder -------------
    def _create_graph_structure(self, bin_data):
        data = HeteroData()

        # 1) Mesh nodes and edges
        data["mesh"].x = _t32(self.mesh_structure["mesh_features_torch"][0])
        data["mesh"].pos = _t32(self.mesh_structure["mesh_lat_lon_list"][0])

        m2m_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
        m2m_edge_attr = self.mesh_structure["m2m_features_torch"][0]
        reverse_edges = torch.stack([m2m_edge_index[1], m2m_edge_index[0]], dim=0)
        data["mesh", "to", "mesh"].edge_index = torch.cat([m2m_edge_index, reverse_edges], dim=1)
        data["mesh", "to", "mesh"].edge_attr = torch.cat([m2m_edge_attr, m2m_edge_attr], dim=0)

        # 2) Observation nodes / edges (supports latent rollout)
        for obs_type, obs_dict in bin_data.items():
            for inst_name, inst_dict in obs_dict.items():
                is_latent_rollout = inst_dict.get("is_latent_rollout", False)

                if is_latent_rollout:
                    # Inputs (shared)
                    if "input_features_final" in inst_dict:
                        node_in = f"{inst_name}_input"
                        data[node_in].x = _t32(inst_dict["input_features_final"])
                        if "input_lat_deg" in inst_dict and "input_lon_deg" in inst_dict:
                            edge_index, edge_attr = obs_mesh_conn(
                                inst_dict["input_lat_deg"],
                                inst_dict["input_lon_deg"],
                                self.mesh_structure["m2m_graphs"],
                                self.mesh_structure["mesh_lat_lon_list"],
                                self.mesh_structure["mesh_list"],
                                o2m=True,
                            )
                            data[node_in, "to", "mesh"].edge_index = edge_index
                            data[node_in, "to", "mesh"].edge_attr = edge_attr

                    # Targets per step
                    steps = int(inst_dict.get("num_latent_steps", 1))
                    for step in range(steps):
                        node_t = f"{inst_name}_target_step{step}"
                        if "target_features_final_list" in inst_dict and step < len(inst_dict["target_features_final_list"]):
                            y = inst_dict["target_features_final_list"][step]
                            data[node_t].y = _t32(y)

                            if "target_metadata_list" in inst_dict and step < len(inst_dict["target_metadata_list"]):
                                data[node_t].target_metadata = _t32(inst_dict["target_metadata_list"][step])

                            if "scan_angle_list" in inst_dict and step < len(inst_dict["scan_angle_list"]):
                                data[node_t].x = _t32(inst_dict["scan_angle_list"][step])

                            if "target_channel_mask_list" in inst_dict and step < len(inst_dict["target_channel_mask_list"]):
                                data[node_t].target_channel_mask = _t32(inst_dict["target_channel_mask_list"][step])

                            if "instrument_id" in inst_dict:
                                data[node_t].instrument_ids = torch.full(
                                    (y.shape[0],), inst_dict["instrument_id"], dtype=torch.long
                                )

                            if ("target_lat_deg_list" in inst_dict and "target_lon_deg_list" in inst_dict
                                and step < len(inst_dict["target_lat_deg_list"])
                                and step < len(inst_dict["target_lon_deg_list"])):
                                lat = inst_dict["target_lat_deg_list"][step]
                                lon = inst_dict["target_lon_deg_list"][step]
                                if len(lat) > 0 and len(lon) > 0:
                                    edge_index, edge_attr = obs_mesh_conn(
                                        lat, lon,
                                        self.mesh_structure["m2m_graphs"],
                                        self.mesh_structure["mesh_lat_lon_list"],
                                        self.mesh_structure["mesh_list"],
                                        o2m=False,
                                    )
                                    data["mesh", "to", node_t].edge_index = edge_index
                                    data["mesh", "to", node_t].edge_attr = edge_attr

                else:
                    node_in = f"{inst_name}_input"
                    node_t = f"{inst_name}_target"

                    if "input_features_final" in inst_dict:
                        data[node_in].x = _t32(inst_dict["input_features_final"])
                        if "input_lat_deg" in inst_dict and "input_lon_deg" in inst_dict:
                            edge_index, edge_attr = obs_mesh_conn(
                                inst_dict["input_lat_deg"],
                                inst_dict["input_lon_deg"],
                                self.mesh_structure["m2m_graphs"],
                                self.mesh_structure["mesh_lat_lon_list"],
                                self.mesh_structure["mesh_list"],
                                o2m=True,
                            )
                            data[node_in, "to", "mesh"].edge_index = edge_index
                            data[node_in, "to", "mesh"].edge_attr = edge_attr

                    if "target_features_final" in inst_dict:
                        y = inst_dict["target_features_final"]
                        data[node_t].y = _t32(y)

                        if "target_metadata" in inst_dict:
                            data[node_t].target_metadata = _t32(inst_dict["target_metadata"])

                        if "scan_angle" in inst_dict:
                            data[node_t].x = _t32(inst_dict["scan_angle"])

                        if "target_channel_mask" in inst_dict:
                            data[node_t].target_channel_mask = _t32(inst_dict["target_channel_mask"])

                        if "instrument_id" in inst_dict:
                            data[node_t].instrument_ids = torch.full((y.shape[0],), inst_dict["instrument_id"], dtype=torch.long)

                        if "target_lat_deg" in inst_dict and "target_lon_deg" in inst_dict:
                            lat = inst_dict["target_lat_deg"]
                            lon = inst_dict["target_lon_deg"]
                            if len(lat) > 0 and len(lon) > 0:
                                edge_index, edge_attr = obs_mesh_conn(
                                    lat, lon,
                                    self.mesh_structure["m2m_graphs"],
                                    self.mesh_structure["mesh_lat_lon_list"],
                                    self.mesh_structure["mesh_list"],
                                    o2m=False,
                                )
                                data["mesh", "to", node_t].edge_index = edge_index
                                data["mesh", "to", node_t].edge_attr = edge_attr

        return data

    # ------------- DataLoaders -------------
    def _worker_init(self, worker_id):
        import numpy as np
        base_seed = int(torch.initial_seed()) % 2**31
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(
            f"[WorkerInit] rank={rank} worker={worker_id} pid={os.getpid()} seed={base_seed} "
            f"train_sum_id={id(self.train_data_summary)} val_sum_id={id(self.val_data_summary)}"
        )

    def train_dataloader(self):
        ds = BinDataset(
            self.train_bin_names,
            self.train_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            tag="TRAIN",
        )
        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,   # safer while debugging stale refs
            worker_init_fn=self._worker_init,
        )
        print(f"[DL] TRAIN v{self._train_version} loader_id={id(loader)} ds_id={id(ds)} sum_id={id(self.train_data_summary)} bins={len(self.train_bin_names)}")
        return loader

    def val_dataloader(self):
        if not self.val_bin_names:
            return None
        ds = BinDataset(
            self.val_bin_names,
            self.val_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            tag="VAL",
        )
        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=self._worker_init,
        )
        print(f"[DL] VAL   v{self._val_version} loader_id={id(loader)} ds_id={id(ds)} sum_id={id(self.val_data_summary)} bins={len(self.val_bin_names)}")
        return loader
