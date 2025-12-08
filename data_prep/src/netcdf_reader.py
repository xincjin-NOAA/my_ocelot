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


def read_netcdf_diag(file_path: str, config: dict = None) -> dict:
    """Read NetCDF diagnostic file and return data as a dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file
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
        nchans = len(ncfile.dimensions['nchans'])
        nobs = len(ncfile.dimensions['nobs'])
        
        print(f"Reading NetCDF file: {file_path}")
        print(f"  nchans: {nchans}, nobs: {nobs}")
        
        # Get variable lists from config
        channel_vars = config.get('channel_vars', [])
        obs_vars = config.get('obs_vars', [])
        
        # Read channel information
        for var_name in channel_vars:
            if var_name in ncfile.variables:
                data[var_name] = ncfile.variables[var_name][:].data
            else:
                print(f"Warning: Variable '{var_name}' not found in NetCDF file")
        
        # Read observation data
        for var_name in obs_vars:
            if var_name in ncfile.variables:
                data[var_name] = ncfile.variables[var_name][:].data
            else:
                print(f"Warning: Variable '{var_name}' not found in NetCDF file")
        
        # Store dimensions
        data['nchans'] = nchans
        data['nobs'] = nobs
        
    return data


def netcdf_to_parquet(input_file: str, output_path: str, date: str = None, cycle: str = None, sat_id: str = None, config: dict = None) -> None:
    """Convert NetCDF diagnostic file to Parquet format.
    
    Parameters
    ----------
    input_file : str
        Path to input NetCDF file
    output_path : str
        Path to output Parquet file or directory
    date : str, optional
        Date string for partitioning (YYYY-MM-DD format)
    cycle : str, optional
        Cycle string for partitioning (e.g., '00', '06', '12', '18')
    sat_id : str, optional
        Satellite ID (e.g., 'n21', 'n20', 'npp')
    config : dict, optional
        Configuration dictionary. If None, loads default config.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Read NetCDF data
    data = read_netcdf_diag(input_file, config)
    
    nchans = data['nchans']
    nobs = data['nobs']
    
    # Calculate number of unique observations (nobs should be divisible by nchans)
    if nobs % nchans != 0:
        raise ValueError(f"nobs ({nobs}) is not divisible by nchans ({nchans})")
    
    n_unique_obs = nobs // nchans
    
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
    
    # Get output variables from config
    output_variables = config.get('output_variables', {})
    
    # Add observation variables (same for all channels, so take every nchans-th value)
    # Assuming data is organized as: [obs1_ch1, obs1_ch2, ..., obs1_chN, obs2_ch1, obs2_ch2, ...]
    for output_name, netcdf_name in output_variables.items():
        var_data = data[netcdf_name][::nchans]
        
        # Handle character arrays (2D) - convert to strings
        if var_data.ndim > 1:
            # Convert character array to string array
            if isinstance(var_data[0], np.ndarray):
                var_data = np.array([''.join(row).strip() for row in var_data])
            else:
                var_data = np.array([str(item).strip() for item in var_data])
        
        table_data[output_name] = pa.array(var_data)
    
    # Reshape observations into separate columns for each channel
    # Reshape from (nobs,) to (n_unique_obs, nchans)
    obs_reshaped = data['Observation'].reshape(n_unique_obs, nchans)
    
    # Create a column for each channel's observation
    for ch_idx in range(nchans):
        # Get the actual channel number from sensor_chan or use index
        chan_num = data['sensor_chan'][ch_idx] if ch_idx < len(data['sensor_chan']) else ch_idx + 1
        table_data[f'observation_ch{chan_num}'] = pa.array(obs_reshaped[:, ch_idx])
    
    # Create table
    table = pa.Table.from_pydict(table_data)
    
    # Write to parquet
    if date is not None:
        # Write as partitioned dataset by date and cycle
        # All satellites for the same date/cycle go into the same partition
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
            existing_table = pq.read_table(partition_dir)
            table = pa.concat_tables([existing_table, table])
            # Remove old partition files before writing
            import shutil
            shutil.rmtree(partition_dir)
        
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=partition_cols,
            existing_data_behavior='overwrite_or_ignore',
            basename_template='part-{i}.parquet'
        )
        
        partition_path = "/".join([f"{col}={date if col == 'date' else cycle}" for col in partition_cols])
        print(f"Written partitioned parquet to: {output_path}/{partition_path}")
    else:
        # Write as single file (append if exists to combine multiple sat_ids)
        if os.path.exists(output_path):
            # Read existing data and append
            existing_table = pq.read_table(output_path)
            table = pa.concat_tables([existing_table, table])
        pq.write_table(table, output_path)
        print(f"Written parquet file to: {output_path}")


def process_netcdf_files(start_date: datetime,
                         end_date: datetime,
                         obs_type: str,
                         base_dir: str,
                         output_path: str,
                         partition_by_date: bool = True,
                         cycles: list = None,
                         sat_ids: list = None,
                         config_path: str = None) -> None:
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
    config_path : str, optional
        Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get cycles from config if not provided
    if cycles is None:
        cycles = config.get('default_cycles', ['00', '06', '12', '18'])
    
    # Get satellite IDs from config if not provided
    if sat_ids is None:
        if obs_type in config['observations']:
            sat_ids = config['observations'][obs_type]['sat_ids']
        else:
            raise ValueError(f"Observation type '{obs_type}' not found in config. "
                           f"Available types: {list(config['observations'].keys())}")
    
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
            # Process each satellite ID
            for sat_id in sat_ids:
                # Construct file path: base_dir/gdas.YYYYMMDD/HH/atmos/diag_{obs_type}_{sat_id}_ges.YYYYMMDDHH.nc4
                date_cycle_str = f"{date_str}{cycle}"
                input_file = os.path.join(
                    base_dir,
                    f"gdas.{date_str}",
                    cycle,
                    "atmos",
                    f"diag_{obs_type}_{sat_id}_ges.{date_cycle_str}.nc4"
                )
                
                if not os.path.exists(input_file):
                    print(f"File not found: {input_file}, skipping...")
                    continue
                
                # Process file
                try:
                    print(f"Processing: {input_file}")
                    netcdf_to_parquet(
                        input_file,
                        output_path,
                        date=date_partition if partition_by_date else None,
                        cycle=cycle if partition_by_date else None,
                        sat_id=sat_id,
                        config=config
                    )
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
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
    output_path = os.path.join(output_path, args.type)
    
    # Get satellite IDs and cycles from config for this obs_type
    sat_ids = None
    if args.type in config.get('observations', {}):
        sat_ids = config['observations'][args.type].get('sat_ids')
    
    cycles = config.get('default_cycles')
    
    # Process files
    process_netcdf_files(
        start_date,
        end_date,
        args.type,
        base_dir,
        output_path,
        partition_by_date=True,
        cycles=cycles,
        sat_ids=sat_ids,
        config_path=args.config
    )
