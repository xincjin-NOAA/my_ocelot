import os
import lightning.pytorch as pl
import pandas as pd
import torch
import torch.distributed as dist
import zarr
import importlib
from nnja_adapter import build_zlike_from_df
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from zarr.storage import LRUStoreCache

from process_timeseries import extract_features, organize_bins_times
from create_mesh_graph_global import obs_mesh_conn


def _t32(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


class BinDataset(Dataset):
    def __init__(
        self,
        bin_names,
        data_summary,
        zarr_store,
        create_graph_fn,
        observation_config,
        feature_stats=None,
        mode="UNKNOWN",
    ):
        self.bin_names = bin_names
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.mode = mode

    def __len__(self):
        return len(self.bin_names)

    def __getitem__(self, idx):
        bin_name = self.bin_names[idx]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        # [PRINT: Data Loading INFO]
        print(f"[Rank {rank}] [{self.mode}] Accessing bin_name: {bin_name}")
        print(f"[Rank {rank}] [{self.mode}] BinDataset.data_summary has {len(self.data_summary)} bins")
        print(f"[Rank {rank}] [{self.mode}] Available bins in BinDataset: \
                {list(self.data_summary.keys())[:5]}...{list(self.data_summary.keys())[-3:] if len(self.data_summary) > 5 else []}")

        if bin_name in self.data_summary:
            print(f"[Rank {rank}] FOUND {bin_name} in BinDataset.data_summary")

            # Check if this is old data or new data
            bin_data = self.data_summary[bin_name]
            if 'satellite' in bin_data and 'atms' in bin_data['satellite']:
                input_time = bin_data['satellite']['atms'].get('input_time')
                print(f"[Rank {rank}] {bin_name} input_time: {input_time}")

        else:
            print(f"[Rank {rank}] ERROR {bin_name} NOT found in BinDataset.data_summary")
            available = list(self.data_summary.keys())
            print(f"[Rank {rank}] Available: {available}")
            raise KeyError(f"Missing {bin_name}")
        # [Loading INFO - END]

        print(f"[Rank {rank}] Fetching {bin_name}...")
        try:
            bin_data = extract_features(self.z, self.data_summary, bin_name, self.observation_config, feature_stats=self.feature_stats)[bin_name]
            graph_data = self.create_graph_fn(bin_data)
            graph_data.bin_name = bin_name
            return graph_data
        except Exception as e:
            print(f"[Rank {rank}] Error processing bin {bin_name}: {e}")
            raise


class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        train_start_date,     # NEW
        train_end_date,       # NEW
        val_start_date,       # NEW
        val_end_date,         # NEW
        train_window_days,    # NEW
        val_window_days,      # NEW
        observation_config,
        mesh_structure,
        batch_size=1,
        num_neighbors=3,
        feature_stats=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_start_date = pd.to_datetime(train_start_date)
        self.train_end_date = pd.to_datetime(train_end_date)
        self.val_start_date = pd.to_datetime(val_start_date)
        self.val_end_date = pd.to_datetime(val_end_date)
        self.train_window_days = pd.Timedelta(days=train_window_days)
        self.val_window_days = pd.Timedelta(days=val_window_days)
        self.train_data_summary = None
        self.val_data_summary = None
        self.mesh_structure = mesh_structure
        self.feature_stats = feature_stats
        self.z = None
        self.data_summary = None
        self.train_bin_names = None
        self.val_bin_names = None

    def setup(self, stage=None):
        if self.z is None:
            self.z = {}
            for obs_type, instruments in self.hparams.observation_config.items():
                self.z[obs_type] = {}
                for inst_name, inst_cfg in instruments.items():
                    src = inst_cfg.get("source", "zarr")

                    if src == "zarr":
                        zarr_dir = inst_cfg.get("zarr_dir")  # full path override (absolute or relative)
                        if zarr_dir:
                            zarr_path = zarr_dir
                        else:
                            zname = inst_cfg.get("zarr_name", inst_name)  # basename or basename.zarr
                            if not zname.endswith(".zarr"):
                                zname += ".zarr"
                            zarr_path = os.path.join(self.hparams.data_path, zname)

                        if not os.path.isdir(zarr_path):
                            raise FileNotFoundError(f"Conventional Zarr not found: {zarr_path}")
                        self.z[obs_type][inst_name] = zarr.open(LRUStoreCache(zarr.DirectoryStore(zarr_path), max_size=2e9), mode="r")
                        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                        if rank == 0:
                            print(f"[ZARR] {obs_type}/{inst_name} -> {zarr_path}")
                            print("  has keys:", list(self.z[obs_type][inst_name].keys())[:12])

                        # Optional guard to ensure the right file is used for surface_obs
                        if obs_type == "conventional" and inst_name == "surface_obs":
                            if not os.path.basename(zarr_path).startswith("raw_surface_obs"):
                                print(f"[WARN] surface_obs expected raw_surface_obs*.zarr but got: {zarr_path}")

                    elif src == "nnja":
                        # Load DataFrame via dotted loader path
                        loader_path = inst_cfg["dataframe_loader"]
                        mod_name, fn_name = loader_path.rsplit(".", 1)
                        load_fn = getattr(importlib.import_module(mod_name), fn_name)

                        # Columns to request = var_map values + coords/time
                        need = list(inst_cfg["var_map"].values())
                        need += [inst_cfg.get("lat_col", "LAT"), inst_cfg.get("lon_col", "LON"), inst_cfg.get("time_col", "OBS_TIMESTAMP")]

                        df = load_fn(start_date=self.hparams.start_date, end_date=self.hparams.end_date, columns=need)

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

        # Calculate dates as pandas objects, then convert to strings for organize_bins_times
        initial_train_start_str = (self.train_start_date).strftime("%Y-%m-%d")
        initial_train_end_str = (self.train_start_date + self.train_window_days).strftime("%Y-%m-%d")

        initial_val_start_str = (self.val_start_date).strftime("%Y-%m-%d")
        initial_val_end_str = (self.val_start_date + self.val_window_days).strftime("%Y-%m-%d")

        # Create train data_summary
        print(f"[TRAIN SETUP] start: {initial_train_start_str}; end: {initial_train_end_str}")
        self.train_data_summary = organize_bins_times(
            self.z,
            initial_train_start_str,
            initial_train_end_str,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
        )

        # Create val data_summary
        print(f"[VAL SETUP] start: {initial_val_start_str}; end: {initial_val_end_str}")
        self.val_data_summary = organize_bins_times(
            self.z,
            initial_val_start_str,
            initial_val_end_str,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
        )

        # Set bin names from respective data summaries
        self.train_bin_names = sorted(list(self.train_data_summary.keys()),
                                      key=lambda x: int(x.replace("bin", "")))
        self.val_bin_names = sorted(list(self.val_data_summary.keys()),
                                    key=lambda x: int(x.replace("bin", "")))

        print(f"[SETUP] train bin names: {self.train_bin_names}")
        print(f"[SETUP] val bin names: {self.val_bin_names}")

        if stage == "fit" or stage is None:
            pass

    def _create_graph_structure(self, bin_data):
        data = HeteroData()

        # 1. Mesh node features and edges
        data["mesh"].x = _t32(self.mesh_structure["mesh_features_torch"][0])
        data["mesh"].pos = _t32(self.mesh_structure["mesh_lat_lon_list"][0])

        # 2. For each instrument, set up input and target nodes and edges
        for obs_type, instruments in self.hparams.observation_config.items():
            for inst_name, inst_cfg in instruments.items():
                node_type_input = f"{inst_name}_input"
                node_type_target = f"{inst_name}_target"

                if inst_name in bin_data.get(obs_type, {}):
                    inst_data = bin_data[obs_type][inst_name]

                    # --- INPUT NODES & EDGES ---
                    data[node_type_input].x = inst_data["input_features_final"]

                    grid_lat_deg = inst_data["input_lat_deg"]
                    grid_lon_deg = inst_data["input_lon_deg"]
                    edge_index_encoder, edge_attr_encoder = obs_mesh_conn(
                        grid_lat_deg,
                        grid_lon_deg,
                        self.mesh_structure["m2m_graphs"],
                        self.mesh_structure["mesh_lat_lon_list"],
                        self.mesh_structure["mesh_list"],
                        o2m=True,
                    )
                    data[node_type_input, "to", "mesh"].edge_index = edge_index_encoder
                    data[node_type_input, "to", "mesh"].edge_attr = edge_attr_encoder

                    # --- TARGET NODES & EDGES ---
                    target_features = inst_data["target_features_final"]  # [N, C]
                    vm = inst_data.get("target_channel_mask", None)
                    if vm is not None:
                        vm = vm.to(torch.bool)

                    keep_t = torch.ones((target_features.shape[0],), dtype=torch.bool) if vm is None else vm.any(dim=1)

                    if keep_t.sum() == 0:
                        data[node_type_target].y = torch.empty((0, target_features.shape[1]), dtype=torch.float32)
                        data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
                        data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
                        data[node_type_target].target_channel_mask = torch.empty((0, target_features.shape[1]), dtype=torch.bool)
                        data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
                        data[node_type_target].pos = torch.empty((0, 2), dtype=torch.float32)
                        data[node_type_target].num_nodes = 0
                    else:
                        keep_np = keep_t.cpu().numpy()
                        y_t = target_features[keep_t]
                        mask_t = torch.ones_like(y_t, dtype=torch.bool) if vm is None else vm[keep_t]
                        tgt_lat = inst_data["target_lat_deg"][keep_np]
                        tgt_lon = inst_data["target_lon_deg"][keep_np]
                        tgt_meta = inst_data["target_metadata"][keep_t]
                        x_aux = (
                            inst_data["scan_angle"][keep_t].to(torch.float32)
                            if inst_name in ("atms", "amsua", "avhrr")
                            else torch.zeros((y_t.shape[0], 1), dtype=torch.float32)
                        )
                        inst_id = int(inst_data["instrument_id"])
                        inst_ids = torch.full((y_t.shape[0],), inst_id, dtype=torch.long)

                        data[node_type_target].y = y_t
                        data[node_type_target].x = x_aux
                        data[node_type_target].instrument_ids = inst_ids
                        data[node_type_target].target_channel_mask = mask_t
                        data[node_type_target].target_metadata = tgt_meta
                        data[node_type_target].pos = torch.stack([_t32(tgt_lon), _t32(tgt_lat)], dim=1)
                        data[node_type_target].num_nodes = y_t.shape[0]

                        edge_index_decoder, edge_attr_decoder = obs_mesh_conn(
                            tgt_lat,
                            tgt_lon,
                            self.mesh_structure["m2m_graphs"],
                            self.mesh_structure["mesh_lat_lon_list"],
                            self.mesh_structure["mesh_list"],
                            o2m=False,
                        )
                        data["mesh", "to", node_type_target].edge_index = edge_index_decoder
                        data["mesh", "to", node_type_target].edge_attr = edge_attr_decoder
                else:
                    # --- Handle missing instruments ---
                    data[node_type_input].x = torch.empty((0, inst_cfg["input_dim"]), dtype=torch.float32)
                    data[node_type_target].y = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.float32)
                    data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
                    data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
                    data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
                    data[node_type_target].target_channel_mask = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.bool)
                    data[node_type_target].pos = torch.empty((0, 2), dtype=torch.float32)
                    data[node_type_target].num_nodes = 0
                    data[node_type_input, "to", "mesh"].edge_index = torch.empty((2, 0), dtype=torch.long)
                    data[node_type_input, "to", "mesh"].edge_attr = torch.empty((0, 3), dtype=torch.float32)
                    data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)  
                    data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 3), dtype=torch.float32)

        # 3. Processor edges (mesh-to-mesh)
        m2m_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
        m2m_edge_attr = self.mesh_structure["m2m_features_torch"][0]

        reverse_edges = torch.stack([m2m_edge_index[1], m2m_edge_index[0]], dim=0)
        data["mesh", "to", "mesh"].edge_index = torch.cat([m2m_edge_index, reverse_edges], dim=1)
        data["mesh", "to", "mesh"].edge_attr = torch.cat([m2m_edge_attr, m2m_edge_attr], dim=0)

        return data

    def set_train_data(self, start_date, end_date):
        """Update train data with new date range"""
        start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date
        end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
        print(f"[UPDATE TRAIN DATA] {start_date} to {end_date}")
        self.train_data_summary = organize_bins_times(
            self.z,
            start_date,
            end_date,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
        )
        self.train_bin_names = sorted(list(self.train_data_summary.keys()),
                                      key=lambda x: int(x.replace("bin", "")))
        print(f"[UPDATE TRAIN] New train bins: {self.train_bin_names}")

    def set_val_data(self, start_date, end_date):
        """Update validation data with new date range"""
        start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else start_date
        end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else end_date
        print(f"[UPDATE VAL DATA] {start_date} to {end_date}")
        self.val_data_summary = organize_bins_times(
            self.z,
            start_date,
            end_date,
            self.hparams.observation_config,
            pipeline_cfg=self.hparams.pipeline,
        )
        self.val_bin_names = sorted(list(self.val_data_summary.keys()),
                                    key=lambda x: int(x.replace("bin", "")))
        print(f"[UPDATE VAL] New val bins: {self.val_bin_names}")

    def train_dataloader(self):
        train_dataset = BinDataset(
            self.train_bin_names,
            self.train_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            mode="TRAIN",
        )
        return PyGDataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self):
        if not self.val_bin_names:
            return None
        val_dataset = BinDataset(
            self.val_bin_names,
            self.val_data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            mode="VAL",
        )
        return PyGDataLoader(
            val_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )
