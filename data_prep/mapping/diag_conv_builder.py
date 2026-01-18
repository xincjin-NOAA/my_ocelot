#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import bufr
import yaml
import faulthandler
import netCDF4 as nc
import re
import sys

from datetime  import datetime, timezone

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
    
config_base = {
    "base_dirs": {
        "input": "/scratch5/purged/Xin.C.Jin/my_ocelot/diag",
        "output": "/scratch3/NCEPDEV/stmp/Xin.C.Jin/temp/ocelot",
    },
    "obs_vars": [
      "Station_ID",
      "Observation_Type",
      "Observation_Subtype",
      "Latitude",
      "Longitude",
      "Model_Elevation",
      "Station_Elevation",
      "Pressure",
      "Height",
      "Time",
      "timestamp",
      "Incremental_Bending_Angle",
      "Analysis_Use_Flag",
      "Prep_Use_Flag",
      "Prep_QC_Mark",
      "Setup_QC_Mark",
      "Observation",
      "u_Observation",
      "v_Observation",
      "u_Obs_Minus_Forecast_adjusted",
      "v_Obs_Minus_Forecast_adjusted",
      "u_Obs_Minus_Forecast_unadjusted",
      "v_Obs_Minus_Forecast_unadjusted",
    ],
}

class ConvDiagObsBuilder(ObsBuilder):
    """
    Conv Diagnostics netCDF reader: reads netCDF files containing Conv diagnostics
    and converts them into BUFR observations according to the mapping defined in diag_conv.yaml.
    """

    def __init__(self, map_dict, log_name=None):
        super().__init__(map_dict, log_name=log_name)
        self.config = config_base
        with open(map_dict, 'r') as f:
            self.type_config = yaml.safe_load(f)
        self.obs_vars = self.config.get('obs_vars', [])
        self.obs_dim_name = self.type_config.get('obs_dim_name', 'nobs')
        self.dim_path_map = {dim["name"]: dim["path"] for dim in self.type_config.get("dimensions", [])}


    def make_obs(self, comm, input_path):
        self.log.debug("***** Entering make_obs *****")
        mapping_path = list(self.map_dict.values())[0]
        # if not self.config:
        #     self.config = load_config(mapping_path)
            
        data = self.netcdf_to_container(input_path, self.config)
        if isinstance(data, dict):
            self.log.debug(f"data keys: {list(data.keys())}")
        else:
            self.log.debug(f"variables in data: {data.list()}")
        return data

    def dims_for_var(self, dims, dim_path_map):
        return [dim_path_map[d] for d in dims]

    def read_netcdf_diag(self,file_path, obs_config) -> dict:

        with nc.Dataset(file_path, 'r') as ncfile:
            # Read dimensions
            nobs = len(ncfile.dimensions[self.obs_dim_name])

            self.log.info(f"Reading NetCDF file: {file_path}")
            data = {}
            # Read channel information
            for var_name in self.obs_vars:
                if var_name in ncfile.variables:
                    data[var_name] = self._maybe_decode_char_array(ncfile.variables[var_name][:])
                else:
                    self.log.debug(f"Warning: Variable '{var_name}' not found in NetCDF file")

            # Store dimensions
            data['nobs'] = nobs
            file_date_str = os.path.basename(file_path).split('.')[-2]
            analysis_time = datetime.strptime(file_date_str, "%Y%m%d%H")
            analysis_time = analysis_time.replace(tzinfo=timezone.utc)
            data["timestamp"] = (analysis_time.timestamp()+ data["Time"].astype(np.float64) * 3600.0).astype(np.int64)
            row_filter = self.type_config.get('row_filter') or self.type_config.get('filter_by_observation_type') 
            if row_filter:
                filter_var = row_filter.get('var')
                filter_values = row_filter.get('values')
                if filter_var and filter_values and filter_var in data:
                    filter_arr = data[filter_var]

                    mask = np.isin(filter_arr, filter_values)

                    if getattr(mask, 'shape', None) and mask.shape[0] == nobs:
                        for k, v in list(data.items()):
                            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nobs:
                                data[k] = v[mask]
                        nobs = int(mask.sum())
        
        # Store dimensions
        data['nobs'] = nobs
        return data

    def get_diag_data(self, input_file: str, config: dict = None):
        # Placeholder for any preprocessing steps needed before reading the NetCDF file
        self.log.info(f"Preparing to read diagnostic data from {input_file}")

        data = self.read_netcdf_diag(input_file, config)
        return data

    def netcdf_to_container(self, input_file: str, date: str = None, cycle: str = None, sat_id: str = None, 
                            config: dict = None, obs_type: str = None):

        data = self.get_diag_data(input_file, config)
       
        nobs = data['nobs']

        # has_channels = nchans is not None and nchans > 0 and nobs % nchans == 0 and 'Observation' in data

        n_unique_obs = nobs 

        use_bufr = False
        if use_bufr:
            container = bufr.BufrContainer()
        else:
            container = {}

        # channel_vars = type_config.get('channel_vars', [])
        # obs_vars = type_config.get('obs_vars', [])
        variables = self.type_config['input']["variables"]

        for name, source in variables.items():
            self.log.debug(source)
            if source in self.obs_vars:
                xr_dims = ['location', ]
                # Reshape observations into separate columns for each channel
                # Reshape from (nobs,) to (n_unique_obs, nchans)
                var_data = data[source]
            else:
                self.log.warning(f"Warning: Skipping variable '{name}' with source '{source}'")
                continue  # Skip variables not in geo_vars or obs_vars
            dim_paths = self.dims_for_var(xr_dims, self.dim_path_map)
        
            self.log.debug(f"  shape =, {var_data.shape}")
            self.log.debug(f"Adding variable '{name}' from source '{source}' with dims {xr_dims} -> paths {dim_paths}")

            if use_bufr:
                container.add_variable(
                    name,
                    var_data,
                    dim_paths
                )
            else:    
                container[name] = var_data 

        return container

    def _maybe_decode_char_array(self, arr):
        if isinstance(arr, np.ma.MaskedArray):
            fill_value = b'' if arr.dtype.kind == 'S' else ''
            arr = arr.filled(fill_value)

        if not isinstance(arr, np.ndarray):
            return arr

        if arr.ndim == 2 and arr.dtype.kind in {'S', 'U'} and arr.dtype.itemsize == 1:
            n_rows, n_cols = arr.shape
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)

            if arr.dtype.kind == 'S':
                row_bytes = arr.view(f'S{n_cols}').reshape(n_rows)
                decoded = np.char.decode(row_bytes, 'utf-8', errors='replace')
                return np.char.strip(decoded)

            row_str = arr.view(f'U{n_cols}').reshape(n_rows)
            return np.char.strip(row_str)

        return arr

