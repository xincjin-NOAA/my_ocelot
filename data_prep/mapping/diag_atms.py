#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import bufr
import yaml
import faulthandler
import netCDF4 as nc

from  netcdf_utils import load_config, netcdf_to_table, read_netcdf_diag
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


MAPPING_PATH = map_path("diag_atms.yaml")


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
        if not self.config:
            self.config = load_config(mapping_path)
            
        data = netcdf_to_table(input_path, self.config)
        return data
    

add_main_functions(AtmsDiagObsBuilder)
