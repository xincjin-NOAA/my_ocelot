import os
import re
from typing import Dict, Union

import pyarrow as pa
import pyarrow.parquet as pq

import bufr
from bufr.obs_builder import add_encoder_type


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
        print(f"CycleParquetEncoder.encode called with date={date}")
        print(f"Container type: {type(container)}")
        print(f"Container keys: {container.keys() if isinstance(container, dict) else 'Not a dict'}")
        
        result: dict = {}
        
        # Collect all tables with cycle information
        all_tables = []
        
        for cycle_key, cycle_container in container.items():
            print(f"Processing cycle: {cycle_key}")
            for category in cycle_container.all_sub_categories():
                substitutions = {}
                for idx, key in enumerate(cycle_container.get_category_map().keys()):
                    substitutions[key] = category[idx]

                dims = self.get_encoder_dimensions(cycle_container, category)
                table = self._build_table(cycle_container, category, dims, cycle_key, date)
                all_tables.append(table)
                result[(cycle_key, tuple(category))] = table
        
        # Combine all tables and write as partitioned dataset
        if all_tables:
            combined_table = pa.concat_tables(all_tables)
            os.makedirs(output_template_path, exist_ok=True)
            
            # Partition by date first, then cycle (order matters for directory structure)
            partition_cols = ['date', 'cycle'] if date is not None else ['cycle']
            
            # Debug: Print partition info
            print(f"Writing partitioned dataset with columns: {partition_cols}")
            print(f"Combined table schema: {combined_table.schema}")
            print(f"Number of rows: {len(combined_table)}")
            if date is not None:
                print(f"Unique dates: {combined_table['date'].unique().to_pylist()}")
            print(f"Unique cycles: {combined_table['cycle'].unique().to_pylist()}")
            
            pq.write_to_dataset(
                combined_table,
                root_path=output_template_path,
                partition_cols=partition_cols,
                existing_data_behavior='overwrite_or_ignore' if append else 'delete_matching',
                basename_template='part-{i}.parquet'
            )

        return result

    def _build_table(
        self,
        container: bufr.DataContainer,
        category: list,
        dims: bufr.encoders.EncoderDimensions,
        cycle: str = None,
        date: str = None,
    ) -> pa.Table:
        """Create a :class:`pyarrow.Table` for the given category."""
        data_dict = {}
        fields = []

        # File level metadata
        file_meta = {
            k.encode(): str(v).encode() for k, v in self.description.get_globals().items()
        }

        # Get number of rows for partition columns
        timestamps = container.get("variables/timestamp", category)
        num_rows = len(timestamps)

        # Track field names to avoid duplicates
        field_names = set()

        # Add date column first (for top-level partitioning)
        if date is not None:
            data_dict["date"] = pa.array([date] * num_rows)
            fields.append(pa.field("date", pa.string()))
            field_names.add("date")
        
        # Add cycle column second (for sub-partitioning under date)
        if cycle is not None:
            data_dict["cycle"] = pa.array([cycle] * num_rows)
            fields.append(pa.field("cycle", pa.string()))
            field_names.add("cycle")

        # Primary time dimension
        timestamps = container.get("variables/timestamp", category)
        data_dict["time"] = pa.array(timestamps)
        fields.append(pa.field("time", data_dict["time"].type))
        field_names.add("time")

        dim_label_map = {d.name().lower(): d.labels for d in dims.dims()}
        
        for var in self.description.get_variables():
            dim_names = [n.lower() for n in dims.dim_names_for_var(var["name"])]
            if not dim_names:
                dim_names = ["time"]
            else:
                dim_names[0] = "time"

            _, var_name = self._split_source_str(var["name"])

            if var_name.lower() in {"datetime", "time"}:
                continue

            source_key = var["source"].split("/")[-1]
            if source_key not in container.list():
                raise ValueError(f"Variable {var['source']} not found in the container")

            var_data = container.get(source_key, category)

            meta = self._field_metadata(var)

            if len(var_data.shape) == 1:
                if var_name not in field_names:
                    array = pa.array(var_data)
                    data_dict[var_name] = array
                    fields.append(pa.field(var_name, array.type, metadata=meta))
                    field_names.add(var_name)
            elif len(var_data.shape) == 2:
                labels = dim_label_map[dim_names[1]]
                for i in range(var_data.shape[1]):
                    col_name = f"{var_name}_{dim_names[1]}_{labels[i]}"
                    if col_name not in field_names:
                        array = pa.array(var_data[:, i])
                        data_dict[col_name] = array
                        fields.append(pa.field(col_name, array.type, metadata=meta))
                        field_names.add(col_name)
            else:
                raise ValueError(
                    f"Variable {var_name} has an invalid shape {var_data.shape}"
                )

        schema = pa.schema(fields, metadata=file_meta)
        return pa.Table.from_pydict(data_dict, schema=schema)

    def _field_metadata(self, var: dict) -> Dict[bytes, bytes]:
        meta: Dict[bytes, bytes] = {
            b"units": str(var["units"]).encode(),
            b"long_name": str(var["longName"]).encode(),
        }
        if "coordinates" in var:
            meta[b"coordinates"] = str(var["coordinates"]).encode()
        if "range" in var:
            meta[b"valid_range"] = str(var["range"]).encode()
        return meta

    def _make_path(self, prototype_path: str, sub_dict: dict) -> str:
        subs = re.findall(r"\{(?P<sub>\w+\/\w+)\}", prototype_path)
        for sub in subs:
            prototype_path = prototype_path.replace(f"{{{sub}}}", sub_dict[sub])
        return prototype_path

    def _split_source_str(self, source: str) -> (str, str):
        components = source.split("/")
        group_name = components[0] if len(components) > 1 else ""
        variable_name = components[-1]
        return group_name, variable_name


add_encoder_type("parquet-ocelot", Encoder)
