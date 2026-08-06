import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc
import yaml

sys.path.insert(0, os.path.realpath('/'))


def load_config(config_path: str = None) -> dict:
    """Load observation configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file. If None, uses default location.
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Default config location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'configs', 'netcdf_obs_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


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
    if config is None:
        config = load_config()
    
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


def netcdf_to_table(input_file: str, date: str = None, cycle: str = None, sat_id: str = None, config: dict = None, obs_type: str = None):
    """Convert NetCDF diagnostic file to PyArrow Table.
    
    Parameters
    ----------
    input_file : str
        Path to input NetCDF file
    date : str, optional
        Date string for partitioning (YYYY-MM-DD format)
    cycle : str, optional
        Cycle string for partitioning (e.g., '00', '06', '12', '18')
    sat_id : str, optional
        Satellite ID (e.g., 'n21', 'n20', 'npp')
    config : dict, optional
        Configuration dictionary. If None, loads default config.
        
    Returns
    -------
    pa.Table
        PyArrow table containing the data
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Extract obs_type from input_file path unless explicitly provided
    if obs_type is None:
        import re
        match = re.search(r'diag_([a-z\-]+)_', os.path.basename(input_file))
        if match:
            obs_type = match.group(1)
        else:
            raise ValueError(f"Could not extract observation type from filename: {input_file}")
    
    # Read NetCDF data
    data = read_netcdf_diag(input_file, obs_type, config)
    
    nchans = data['nchans']
    nobs = data['nobs']

    has_channels = nchans is not None and nchans > 0 and nobs % nchans == 0 and 'Observation' in data
    if has_channels:
        n_unique_obs = nobs // nchans
    else:
        n_unique_obs = nobs
    
    if has_channels:
        print(f"Reshaping data: {nobs} total obs -> {n_unique_obs} unique obs x {nchans} channels")
    
    # Create PyArrow table from observation data
    # Each row represents one unique observation with all channels
    table_data = {}
    
    # Add date column if provided (for partitioning)
    if date is not None:
        table_data['date'] = pa.array([date] * n_unique_obs)
    
    # Add cycle column if provided (for partitioning)
    if cycle is not None:
        table_data['cycle'] = pa.array([cycle] * n_unique_obs)
    
    # Add sat_id column if provided
    if sat_id is not None:
        table_data['sat_id'] = pa.array([sat_id] * n_unique_obs)
    
    # Get output variables from observation-specific config
    obs_config = config.get('observations', {}).get(obs_type, {})
    output_variables_cfg = obs_config.get('output_variables', {})
    if isinstance(output_variables_cfg, list):
        output_variables = {v: v for v in output_variables_cfg}
    else:
        output_variables = output_variables_cfg or {}

    if not output_variables:
        for var_name, var_data in data.items():
            if var_name in {'nchans', 'nobs', 'obs_dim_name'}:
                continue
            if isinstance(var_data, np.ndarray) and var_data.ndim >= 1 and var_data.shape[0] == n_unique_obs:
                output_variables[var_name] = var_name

    output_variables_lc = {}
    for output_name, netcdf_name in output_variables.items():
        output_name_lc = output_name.lower()
        if output_name_lc in output_variables_lc and output_variables_lc[output_name_lc] != netcdf_name:
            raise ValueError(
                "Output variable name collision after lowercasing: "
                f"'{output_name}' and another variable both map to '{output_name_lc}'."
            )
        output_variables_lc[output_name_lc] = netcdf_name
    output_variables = output_variables_lc
    
    # Add observation variables (same for all channels, so take every nchans-th value)
    # Assuming data is organized as: [obs1_ch1, obs1_ch2, ..., obs1_chN, obs2_ch1, obs2_ch2, ...]
    for output_name, netcdf_name in output_variables.items():
        if netcdf_name not in data:
            continue

        if has_channels:
            var_data = data[netcdf_name][::nchans]
        else:
            var_data = data[netcdf_name]
        
        # Handle character arrays (2D) - convert to strings
        if isinstance(var_data, np.ndarray) and var_data.ndim > 1:
            # Convert character array to string array
            if isinstance(var_data[0], np.ndarray):
                var_data = np.array([''.join(row).strip() for row in var_data])
            else:
                var_data = np.array([str(item).strip() for item in var_data])
        
        table_data[output_name] = pa.array(var_data)

    if has_channels:
        # Reshape observations into separate columns for each channel
        # Reshape from (nobs,) to (n_unique_obs, nchans)
        obs_reshaped = data['Observation'].reshape(n_unique_obs, nchans)
        
        # Create a column for each channel's observation
        sensor_chan = data.get('sensor_chan')
        for ch_idx in range(nchans):
            if isinstance(sensor_chan, np.ndarray) and ch_idx < len(sensor_chan):
                chan_num = sensor_chan[ch_idx]
            else:
                chan_num = ch_idx + 1
            table_data[f'observation_ch{chan_num}'] = pa.array(obs_reshaped[:, ch_idx])
    
    # Create table
    table = pa.Table.from_pydict(table_data)
    return table


def write_tables_to_parquet(tables: list, output_path: str, date: str = None, cycle: str = None) -> None:
    """Write multiple PyArrow tables to Parquet format.
    
    Parameters
    ----------
    tables : list
        List of PyArrow tables to write
    output_path : str
        Path to output Parquet file or directory
    date : str, optional
        Date string for partitioning (YYYY-MM-DD format)
    cycle : str, optional
        Cycle string for partitioning (e.g., '00', '06', '12', '18')
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    if not tables:
        print("No tables to write")
        return
    
    # Concatenate all tables
    print(f"Concatenating {len(tables)} tables...")
    try:
        combined_table = pa.concat_tables(tables, promote_options='permissive')
    except Exception as e:
        print(f"Warning: Schema mismatch when concatenating. Using pandas fallback.")
        print(f"Error: {e}")
        import pandas as pd
        dfs = [table.to_pandas() for table in tables]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_table = pa.Table.from_pandas(combined_df)
    
    # Write to parquet
    if date is not None:
        # Write as partitioned dataset by date and cycle
        os.makedirs(output_path, exist_ok=True)
        
        # Determine partition columns
        partition_cols = ['date']
        if cycle is not None:
            partition_cols.append('cycle')
        
        # Read existing data in this partition if it exists
        if cycle is not None:
            partition_dir = os.path.join(output_path, f"date={date}", f"cycle={cycle}")
        else:
            partition_dir = os.path.join(output_path, f"date={date}")
        
        if os.path.exists(partition_dir):
            # Read existing data and append
            print(f"Found existing partition at {partition_dir}, appending...")
            existing_table = pq.read_table(partition_dir)
            
            # Ensure schemas are compatible by promoting to a unified schema
            try:
                combined_table = pa.concat_tables([existing_table, combined_table], promote_options='permissive')
            except Exception as e:
                print(f"Warning: Schema mismatch when appending. Existing schema: {existing_table.schema}")
                print(f"New schema: {combined_table.schema}")
                print(f"Error: {e}")
                # Try to unify schemas by converting to pandas and back
                import pandas as pd
                df_existing = existing_table.to_pandas()
                df_new = combined_table.to_pandas()
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                combined_table = pa.Table.from_pandas(df_combined)
            
            # Remove old partition files before writing
            import shutil
            shutil.rmtree(partition_dir)
        
        pq.write_to_dataset(
            combined_table,
            root_path=output_path,
            partition_cols=partition_cols,
            existing_data_behavior='overwrite_or_ignore',
            basename_template='part-{i}.parquet'
        )
        
        partition_path = "/".join([f"{col}={date if col == 'date' else cycle}" for col in partition_cols])
        print(f"Written {len(combined_table)} rows to partitioned parquet: {output_path}/{partition_path}")
    else:
        # Write as single file
        if os.path.exists(output_path):
            # Read existing data and append
            print(f"Found existing file at {output_path}, appending...")
            existing_table = pq.read_table(output_path)
            
            # Ensure schemas are compatible
            try:
                combined_table = pa.concat_tables([existing_table, combined_table], promote_options='permissive')
            except Exception as e:
                print(f"Warning: Schema mismatch when appending. Using pandas fallback.")
                import pandas as pd
                df_existing = existing_table.to_pandas()
                df_new = combined_table.to_pandas()
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                combined_table = pa.Table.from_pandas(df_combined)
        
        pq.write_table(combined_table, output_path)
        print(f"Written {len(combined_table)} rows to parquet file: {output_path}")


def process_netcdf_files(start_date: datetime,
                         end_date: datetime,
                         obs_type: str,
                         base_dir: str,
                         output_path: str,
                         config: dict,
                         partition_by_date: bool = True,
                         cycles: list = None,
                         sat_ids: list = None) -> None:
    """Process multiple NetCDF files and convert to Parquet.
    
    Parameters
    ----------
    start_date : datetime
        Start date for processing
    end_date : datetime
        End date for processing
    obs_type : str
        Type of observation (e.g., 'atms', 'cris', 'amsua')
    base_dir : str
        Base directory containing gdas.YYYYMMDD subdirectories
        Example: '/path/to/data'
        Expected structure: base_dir/gdas.YYYYMMDD/HH/atmos/diag_{obs_type}_{sat_id}_ges.YYYYMMDDHH.nc4
    output_path : str
        Output directory for parquet files
    partition_by_date : bool
        Whether to partition output by date
    cycles : list, optional
        List of cycles to process (e.g., ['00', '06', '12', '18'])
        If None, uses default from config
    sat_ids : list, optional
        List of satellite IDs to process (e.g., ['n20', 'n21', 'npp'])
        If None, uses all satellite IDs from config for the given obs_type
    config : dict
        Pre-loaded configuration dictionary
    """
    if config is None:
        raise ValueError("config must be provided (pre-loaded) to process_netcdf_files")

    # Get cycles from config if not provided
    if cycles is None:
        cycles = config.get('default_cycles', ['00', '06', '12', '18'])

    obs_cfg = config.get('observations', {}).get(obs_type)
    if obs_cfg is None:
        raise ValueError(
            f"Observation type '{obs_type}' not found in config. "
            f"Available types: {list(config.get('observations', {}).keys())}"
        )

    file_obs_type = obs_cfg.get('input_src') or obs_type

    # Get satellite IDs from config if not provided
    if sat_ids is None:
        sat_ids = obs_cfg.get('sat_ids')
    
    # Allow obs types that do not have a satellite dimension
    if not sat_ids:
        sat_ids = [None]
    
    print(f"Processing observation type: {obs_type}")
    print(f"Satellite IDs: {sat_ids}")
    print(f"Cycles: {cycles}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    day = timedelta(days=1)
    date = start_date
    
    while date <= end_date:
        date_str = date.strftime("%Y%m%d")
        date_partition = date.strftime("%Y-%m-%d")
        
        # Process each cycle for this date
        for cycle in cycles:
            # Collect all tables for this date/cycle combination
            tables_for_cycle = []
            
            # Process each satellite ID
            for sat_id in sat_ids:
                # Construct file path: base_dir/gdas.YYYYMMDD/HH/atmos/diag_{obs_type}_{sat_id}_ges.YYYYMMDDHH.nc4
                date_cycle_str = f"{date_str}{cycle}"
                if sat_id is None:
                    file_name = f"diag_{file_obs_type}_ges.{date_cycle_str}.nc4"
                else:
                    file_name = f"diag_{file_obs_type}_{sat_id}_ges.{date_cycle_str}.nc4"
	
                input_file = os.path.join(
                    base_dir,
                    f"gdas.{date_str}",
                    cycle,
                    "atmos",
                    file_name
                )
                
                if not os.path.exists(input_file):
                    print(f"File not found: {input_file}, skipping...")
                    continue
                
                # Process file to table
                try:
                    print(f"Processing: {input_file}")
                    table = netcdf_to_table(
                        input_file,
                        date=date_partition if partition_by_date else None,
                        cycle=cycle if partition_by_date else None,
                        sat_id=sat_id,
                        config=config,
                        obs_type=obs_type,
                    )
                    tables_for_cycle.append(table)
                    print(f"  -> Created table with {len(table)} rows for {sat_id}")
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Write all tables for this cycle at once
            if tables_for_cycle:
                try:
                    print(f"\nWriting {len(tables_for_cycle)} tables for {date_partition} cycle {cycle}...")
                    write_tables_to_parquet(
                        tables_for_cycle,
                        output_path,
                        date=date_partition if partition_by_date else None,
                        cycle=cycle if partition_by_date else None
                    )
                    print()
                except Exception as e:
                    print(f"Error writing tables for {date_partition} cycle {cycle}: {e}")
                    import traceback
                    traceback.print_exc()
        
        date += day


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert NetCDF diagnostic files to Parquet format')
    parser.add_argument('start_date', help='Start date in YYYY-MM-DD format')
    parser.add_argument('end_date', help='End date in YYYY-MM-DD format')
    parser.add_argument('type', help='Observation type (e.g., atms, cris)')
    parser.add_argument('output_type', choices=['parquet'], help='Output file type (only parquet supported)')
    parser.add_argument('-s', '--suffix', required=False, help='Suffix for the output file(s) (optional, not used)')
    parser.add_argument('-a', '--append', action='store_true', help='Append to existing data')
    parser.add_argument('--config', help='Path to configuration YAML file')
    parser.add_argument('--cycles', help='Comma-separated list of cycles to process (e.g., 00,06,12,18). Overrides config default_cycles.')
    parser.add_argument('--sat-ids', help='Comma-separated list of satellite IDs to process (e.g., n20,n21,npp). Overrides config observations.<type>.sat_ids.')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Load config to get paths and settings
    config = load_config(args.config)
    
    # Get base directories from config
    base_dir = config.get('base_dirs', {}).get('input')
    output_path = config.get('base_dirs', {}).get('output')
    
    if not base_dir:
        raise ValueError("Input base directory not configured in config file under base_dirs.input")
    if not output_path:
        raise ValueError("Output base directory not configured in config file under base_dirs.output")
    
    # Create output directory with obs_type subdirectory
    output_path = os.path.join(output_path, args.type + ".parquet")

    cycles = [c.strip() for c in args.cycles.split(',') if c.strip()] if args.cycles else None
    sat_ids = [s.strip() for s in args.sat_ids.split(',') if s.strip()] if args.sat_ids else None

    process_kwargs = {
        'partition_by_date': True,
    }
    if cycles is not None:
        process_kwargs['cycles'] = cycles
    if sat_ids is not None:
        process_kwargs['sat_ids'] = sat_ids
    
    # Process files
    process_netcdf_files(
        start_date,
        end_date,
        args.type,
        base_dir,
        output_path,
        config,
        **process_kwargs,
    )
