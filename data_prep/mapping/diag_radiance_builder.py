#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import bufr
import yaml
import faulthandler
import netCDF4 as nc
import sys

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path("diag_atms.yaml")
    

config_base = {
    "base_dirs": {
        "input": "/scratch5/purged/Xin.C.Jin/my_ocelot/diag",
        "output": "/scratch3/NCEPDEV/stmp/Xin.C.Jin/temp/ocelot",
    },
    "channel_vars": [
        "chaninfoidx",
        "frequency",
        "polarization",
        "wavenumber",
        "error_variance",
        "mean_lapse_rate",
        "use_flag",
        "sensor_chan",
        "satinfo_chan",
    ],
    "geo_vars": [
        "Channel_Index",
        "Observation_Class",
        "Latitude",
        "Longitude",
        "Elevation",
        "Obs_Time",
        "Scan_Position",
        "Sat_Zenith_Angle",
        "Sat_Azimuth_Angle",
        "Sol_Zenith_Angle",
        "Sol_Azimuth_Angle",
        "Sun_Glint_Angle",
        "Scan_Angle",
        "Water_Fraction",
    ],
    "obs_vars": [
        "Observation",
    ],
}


class RadianceDiagObsBuilder(ObsBuilder):
    """
    Atms Diagnostics netCDF reader: reads netCDF files containing ATMS diagnostics
    and converts them into BUFR observations according to the mapping defined in diag_atms.yaml.
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))
        self.config = config_base
        with open(MAPPING_PATH, 'r') as f:
            self.type_config = yaml.safe_load(f)
        self.channel_vars = self.config.get('channel_vars', [])
        self.obs_vars = self.config.get('obs_vars', [])
        self.geo_vars = self.config.get('geo_vars', [])
        self.obs_dim_name = self.type_config.get('obs_dim_name', 'nobs')
        self.dim_path_map = {dim["name"]: dim["path"] for dim in self.type_config.get("dimensions", [])}

    def make_obs(self, comm, input_path):
        print("***** Entering make_obs *****")
        mapping_path = list(self.map_dict.values())[0]
        # if not self.config:
        #     self.config = load_config(mapping_path)
            
        data = self.netcdf_to_container(input_path, self.config)
        print(f"variables in data: {data.list()}")
        
        return data

    def dims_for_var(self, dims, dim_path_map):
        return [dim_path_map[d] for d in dims]

    def read_netcdf_diag(self,file_path, obs_config) -> dict:

        with nc.Dataset(file_path, 'r') as ncfile:
            # Read dimensions
            nchans = len(ncfile.dimensions['nchans']) if 'nchans' in ncfile.dimensions else None

            nobs = len(ncfile.dimensions[self.obs_dim_name])

            print(f"Reading NetCDF file: {file_path}")
            print(f"  nchans: {nchans}, {self.obs_dim_name}: {nobs}")
            data = {}
            # Read channel information
            for var_name in self.channel_vars + self.geo_vars + self.obs_vars:
                if var_name in ncfile.variables:
                    data[var_name] = _maybe_decode_char_array(ncfile.variables[var_name][:])
                else:
                    print(f"Warning: Variable '{var_name}' not found in NetCDF file")

            # Store dimensions
            data['nchans'] = nchans
            data['nobs'] = nobs

        return data

    def get_diag_data(self, input_file: str, config: dict = None):
        # Placeholder for any preprocessing steps needed before reading the NetCDF file
        print(f"Preparing to read diagnostic data from {input_file}")
        data = self.read_netcdf_diag(input_file, config)
        return data

    def netcdf_to_container(self, input_file: str, date: str = None, cycle: str = None, sat_id: str = None, 
                            config: dict = None, obs_type: str = None):

        data = self.get_diag_data(input_file, config)
       
        nchans = data['nchans']
        nobs = data['nobs']

        # has_channels = nchans is not None and nchans > 0 and nobs % nchans == 0 and 'Observation' in data

        n_unique_obs = nobs // nchans
        print(f"Reshaping data: {nobs} total obs -> {n_unique_obs} unique obs x {nchans} channels")

        container = bufr.DataContainer()

        # channel_vars = type_config.get('channel_vars', [])
        # obs_vars = type_config.get('obs_vars', [])
        variables = self.type_config['input']["variables"]

        for var in variables:
            name = var["name"]
            source = var["source"]
            print(source)
            if source in self.obs_vars:
                xr_dims = ['location', 'channel']
                # Reshape observations into separate columns for each channel
                # Reshape from (nobs,) to (n_unique_obs, nchans)
                var_data = data[source].reshape(n_unique_obs, nchans)

            elif source in self.geo_vars:
                xr_dims = ['location']
                var_data = data[source][::nchans]
            else:
                continue  # Skip variables not in geo_vars or obs_vars
            dim_paths = self.dims_for_var(xr_dims, self.dim_path_map)
        
            print("  shape =", var_data.shape)
            print(f"Adding variable '{name}' from source '{source}' with dims {xr_dims} -> paths {dim_paths}")
            container.add(
                name,
                data[source],
                dim_paths
            )
        return container

    
def _maybe_decode_char_array(arr):
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

    def _maybe_decode_char_array(arr):
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

add_main_functions(RadianceDiagObsBuilder)