import os
import re
from typing import Dict, Union

import pyarrow as pa
import pyarrow.parquet as pq

import bufr
from bufr.obs_builder import add_encoder_type
from netcdf_utils import write_tables_to_parquet

# Encoder for Apache Parquet format
class Encoder(bufr.encoders.EncoderBase):
    def __init__(self, description: Union[str, bufr.encoders.Description]):
        if isinstance(description, str):
            self.description = bufr.encoders.Description(description)
        else:
            self.description = description
        super().__init__(self.description)

    def encode(
        self,
        container: dict,
        output_template_path: str,
        append: bool = False,
        date: str = None,
    ) -> dict:
        """Encode the DataContainer dict into partitioned parquet files.

        Parameters
        ----------
        container: dict
            Dictionary mapping cycle keys to DataContainer objects.
        output_template_path: str
            Path template where the parquet files will be written. This will be
            used as the root directory for partitioned output.
        append: bool, optional
            If ``True`` the encoded data will be appended to an existing parquet
            dataset, otherwise a new dataset will be created or overwrite an
            existing one.
        date: str, optional
            If provided, adds a date column and partitions by date first, then cycle.
        """
        write_tables_to_parquet(
            container,
            output_template_path,
            date=date
        )