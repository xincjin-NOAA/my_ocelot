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

     def read_netcdf_diag(self, file_path, obs_config) -> dict:
        self.log.info(f"Reading GRIB file: {file_path}")

        datasets = xr.open_dataset(file_path, engine='cfgrib',
                                    backend_kwargs={'indexpath': ''})
        ds = xr.merge(datasets)

        time_name = next((c for c in ('time', 'valid_time') if c in ds.coords), None)
        x_name    = next((c for c in ('x', 'longitude') if c in ds.coords), None)
        y_name    = next((c for c in ('y', 'latitude')  if c in ds.coords), None)

        if not all([time_name, x_name, y_name]):
            raise ValueError(
                f"Could not resolve time/x/y coords in GRIB file {file_path}. "
                f"Available coords: {list(ds.coords)}"
            )

        time_vals = ds[time_name].values
        y_vals    = ds[y_name].values
        x_vals    = ds[x_name].values

        ntime = time_vals.shape[0] if time_vals.ndim > 0 else 1
        ny    = y_vals.shape[0]
        nx    = x_vals.shape[-1] if x_vals.ndim > 1 else x_vals.shape[0]
        nobs  = ntime * ny * nx

        t_grid = np.repeat(time_vals, ny * nx)
        if x_vals.ndim == 1 and y_vals.ndim == 1:
            yy, xx = np.meshgrid(y_vals, x_vals, indexing='ij')
        else:
            yy, xx = y_vals, x_vals
        y_flat = np.tile(yy.ravel(), ntime)
        x_flat = np.tile(xx.ravel(), ntime)

        data = {y_name: y_flat, x_name: x_flat}

        for var_name in self.obs_vars:
            if var_name in ds:
                data[var_name] = ds[var_name].values.reshape(-1)
            else:
                self.log.debug(f"Warning: Variable '{var_name}' not found in GRIB file")

        data['nobs'] = nobs
        data['timestamp'] = t_grid.astype('datetime64[s]').astype(np.int64)

        row_filter = self.type_config.get('row_filter') or self.type_config.get('filter_by_observation_type')
        if row_filter:
            filter_var    = row_filter.get('var')
            filter_values = row_filter.get('values')
            if filter_var and filter_values and filter_var in data:
                mask = np.isin(data[filter_var], filter_values)
                if getattr(mask, 'shape', None) and mask.shape[0] == nobs:
                    for k, v in list(data.items()):
                        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nobs:
                            data[k] = v[mask]
                    nobs = int(mask.sum())

        data['nobs'] = nobs
        return data

