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
    "observations": {
        "atms": {
            "sat_ids": ["n20", "n21", "npp"],
            "description": "Advanced Technology Microwave Sounder",
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
            "obs_vars": [
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
                "Observation",
            ],
            "output_variables": {
                "latitude": "Latitude",
                "longitude": "Longitude",
                "obs_time": "Obs_Time",
                "scan_position": "Scan_Position",
                "sat_zenith_angle": "Sat_Zenith_Angle",
                "sat_azimuth_angle": "Sat_Azimuth_Angle",
                "sol_zenith_angle": "Sol_Zenith_Angle",
                "sol_azimuth_angle": "Sol_Azimuth_Angle",
                "sun_glint_angle": "Sun_Glint_Angle",
                "scan_angle": "Scan_Angle",
            },
        }
    },
    "default_cycles": ["00", "06", "12", "18"],
    "file_pattern": "diag_{obs_type}_{sat_id}_ges.{date_cycle}.nc4",
    "dir_pattern": "gdas.{date}/{cycle}/atmos",
}



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


def read_netcdf_diag(file_path: str, obs_type: str, config: dict = None) -> dict:
    """Read NetCDF diagnostic file and return data as a dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file
    obs_type : str
        Observation type (e.g., 'atms', 'cris')
    config : dict, optional
        Configuration dictionary. If None, loads default config.
        
    Returns
    -------
    dict
        Dictionary containing all variables from the NetCDF file
    """

    if config == None:
        config = config_base
    data = {}
    
    with nc.Dataset(file_path, 'r') as ncfile:
        # Read dimensions
        nchans = len(ncfile.dimensions['nchans']) if 'nchans' in ncfile.dimensions else None

        obs_dim_name = None
        for candidate in ['nobs', 'nlocs', 'nrecs', 'nprofiles']:
            if candidate in ncfile.dimensions:
                obs_dim_name = candidate
                break

        if obs_dim_name is None:
            raise ValueError(f"Could not determine observation dimension in NetCDF file: {file_path}")

        nobs = len(ncfile.dimensions[obs_dim_name])
        
        print(f"Reading NetCDF file: {file_path}")
        if nchans is not None:
            print(f"  nchans: {nchans}, {obs_dim_name}: {nobs}")
        else:
            print(f"  {obs_dim_name}: {nobs}")
        
        # Get variable lists from observation-specific config
        obs_config = config.get('observations', {}).get(obs_type, {})
        channel_vars = obs_config.get('channel_vars', [])
        obs_vars = obs_config.get('obs_vars', [])

        if not channel_vars and not obs_vars:
            obs_vars = list(ncfile.variables.keys())
        
        # Read channel information
        if nchans is not None:
            for var_name in channel_vars:
                if var_name in ncfile.variables:
                    data[var_name] = _maybe_decode_char_array(ncfile.variables[var_name][:])
                else:
                    print(f"Warning: Variable '{var_name}' not found in NetCDF file")
        
        # Read observation data
        for var_name in obs_vars:
            if var_name in ncfile.variables:
                data[var_name] = _maybe_decode_char_array(ncfile.variables[var_name][:])
            else:
                print(f"Warning: Variable '{var_name}' not found in NetCDF file")

        row_filter = obs_config.get('row_filter') or obs_config.get('filter_by_observation_type')
        if row_filter:
            filter_var = row_filter.get('var')
            filter_values = row_filter.get('values')
            if filter_var and filter_values and filter_var in data:
                filter_arr = data[filter_var]

                if nchans is not None and nobs % nchans == 0 and getattr(filter_arr, 'shape', None) and filter_arr.shape[0] == (nobs // nchans):
                    mask_unique = np.isin(filter_arr, filter_values)
                    mask = np.repeat(mask_unique, nchans)
                else:
                    mask = np.isin(filter_arr, filter_values)

                if getattr(mask, 'shape', None) and mask.shape[0] == nobs:
                    for k, v in list(data.items()):
                        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nobs:
                            data[k] = v[mask]
                    nobs = int(mask.sum())
        
        # Store dimensions
        data['nchans'] = nchans
        data['nobs'] = nobs
        data['obs_dim_name'] = obs_dim_name
        
    return data


def dims_for_var(varname, dims, dim_path_map):
        """
        Map xarray dimension names (e.g. ('location', 'npc_global'))
        to BUFR query strings using the 'dimensions' section in cris_pca.yaml.
        """
        dim_paths = []
        for d in dims:
            if d not in dim_path_map:
                raise RuntimeError(
                    f"_dims_for_var: no mapping for dimension '{d}' "
                    f"in encoder YAML; known: "
                    f"{list(dim_path_map.keys())}"
                )
            dim_paths.append(dim_path_map[d])

        return dim_paths
        

def netcdf_to_container(input_file: str, date: str = None, cycle: str = None, sat_id: str = None, config: dict = None, obs_type: str = None):

    if config == None:
        config = config_base
    obs_type = 'atms'
    file_path = input_file
    
    # Get variable lists from observation-specific config
    obs_config = config.get('observations', {}).get(obs_type, {})
    print(obs_config.keys())
    channel_vars = obs_config.get('channel_vars', [])
    obs_vars = obs_config.get('obs_vars', [])
    
    
    # Extract obs_type from input_file path unless explicitly provided
    if obs_type is None:
        import re
        match = re.search(r'diag_([a-z\-]+)_', os.path.basename(file_path))
        if match:
            obs_type = match.group(1)
        else:
            raise ValueError(f"Could not extract observation type from filename: {file_path}")
    
    # Read NetCDF data
    data = read_netcdf_diag(file_path, obs_type, config)
    
    nchans = data['nchans']
    nobs = data['nobs']
    
    # has_channels = nchans is not None and nchans > 0 and nobs % nchans == 0 and 'Observation' in data
    has_channels = True
    
    if has_channels:
        n_unique_obs = nobs // nchans
    else:
        n_unique_obs = nobs
    
    if has_channels:
        print(f"Reshaping data: {nobs} total obs -> {n_unique_obs} unique obs x {nchans} channels")
    
    yaml_path = '/scratch3/NCEPDEV/da/Xin.C.Jin/git/my_ocelot/data_prep/mapping/diag_atms.yaml'
    type_config = yaml.load(open(yaml_path), Loader=yaml.Loader)
    
    
    container = bufr.DataContainer()
    dim_path_map = {}
    for dim in type_config.get("dimensions", []):
        n = dim["name"]
        p = dim["path"]
        dim_path_map[n] = p
    
    
    print(dim_path_map)
    # channel_vars = type_config.get('channel_vars', [])
    # obs_vars = type_config.get('obs_vars', [])
    variables = type_config['input']["variables"]
    print(obs_vars)
    print(channel_vars)
    print(variables)
    
    for var in variables:
        name = var["name"]
        source = var["source"]
        print(source)
        if source in ['Observation',]:
            xr_dims = ['location', 'channel']
            if has_channels:
                # Reshape observations into separate columns for each channel
                # Reshape from (nobs,) to (n_unique_obs, nchans)
                var_data = data[source].reshape(n_unique_obs, nchans)
            else:
                var_data = data[source]
        elif source in obs_vars:
            xr_dims = ['location']
            if has_channels:
                var_data = data[source][::nchans]
            else:
                var_data = data[source]
        else:
            xr_dims = ['location']
            var_data = data[source]
        dim_paths = dims_for_var(name, xr_dims, dim_path_map)
    
        print("  shape =", var_data.shape)
        print(f"Adding variable '{name}' from source '{source}' with dims {xr_dims} -> paths {dim_paths}")
        container.add(
            name,
            data[source],
            dim_paths
        )
    return container


class AtmsDiagObsBuilder(ObsBuilder):
    """
    Atms Diagnostics netCDF reader: reads netCDF files containing ATMS diagnostics
    and converts them into BUFR observations according to the mapping defined in diag_atms.yaml.
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        print("***** Entering make_obs *****")
        mapping_path = list(self.map_dict.values())[0]
        # if not self.config:
        #     self.config = load_config(mapping_path)
            
        data = netcdf_to_container(input_path, self.config)
        print(f"variables in data: {data.list()}")
        
        return data

add_main_functions(AtmsDiagObsBuilder)