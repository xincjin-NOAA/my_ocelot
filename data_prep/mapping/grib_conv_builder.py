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
    
base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.realpath(os.path.join(base_path, '..', 'mapping')))

from diag_conv_builder import ConvDiagObsBuilder, config_base


class ConvGribObsBuilder(ConvDiagObsBuilder):

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
            self.log.debug(f"file_date_str: {file_date_str}")
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

