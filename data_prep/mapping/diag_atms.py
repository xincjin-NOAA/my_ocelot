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

from bufr.obs_builder import  add_main_functions, map_path

base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.realpath(os.path.join(base_path, '..', 'mapping')))
os.environ["LOG_LEVEL"] = "DEBUG"
from diag_radiance_builder import RadianceDiagObsBuilder

MAPPING_PATH = map_path("diag_atms.yaml")


class AtmsDiagObsBuilder(RadianceDiagObsBuilder):
    """
    Atms Diagnostics netCDF reader: reads netCDF files containing ATMS diagnostics
    and converts them into BUFR observations according to the mapping defined in diag_atms.yaml.
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))
        # self.log.set_level("DEBUG")

    def make_obs(self, comm, input_path):
        self.log.debug("***** Entering make_obs *****")
        mapping_path = list(self.map_dict.values())[0]
        # if not self.config:
        #     self.config = load_config(mapping_path)
            
        data = self.netcdf_to_container(input_path, self.config)
        self.log.debug(f"variables in data: {data.list()}")
        
        return data

add_main_functions(AtmsDiagObsBuilder)