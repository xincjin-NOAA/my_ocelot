import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.realpath('/'))
from parquet_encoder import Encoder as ParquetEncoder  # noqa: E402
import settings  # noqa: E402


def read_netcdf_diag(file_path: str) -> dict:
    """Read NetCDF diagnostic file and return data as a dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file
        
    Returns
    -------
    dict
        Dictionary containing all variables from the NetCDF file
    """
    data = {}
    
    with nc.Dataset(file_path, 'r') as ncfile:
        # Read dimensions
        nchans = len(ncfile.dimensions['nchans'])
        nobs = len(ncfile.dimensions['nobs'])
        
        print(f"Reading NetCDF file: {file_path}")
        print(f"  nchans: {nchans}, nobs: {nobs}")
        
        # Define channel information variables (dimension: nchans)
        channel_vars = [
            'chaninfoidx',
            'frequency',
            'polarization',
            'wavenumber',
            'error_variance',
            'mean_lapse_rate',
            'use_flag',
            'sensor_chan',
            'satinfo_chan',
        ]
        
        # Read channel information
        for var_name in channel_vars:
            if var_name in ncfile.variables:
                data[var_name] = ncfile.variables[var_name][:].data
            else:
                print(f"Warning: Variable '{var_name}' not found in NetCDF file")
        
        # Define observation data variables (dimension: nobs)
        obs_vars = [
            'Channel_Index',
            'Observation_Class',
            'Latitude',
            'Longitude',
            'Elevation',
            'Obs_Time',
            'Scan_Position',
            'Sat_Zenith_Angle',
            'Sat_Azimuth_Angle',
            'Sol_Zenith_Angle',
            'Sol_Azimuth_Angle',
            'Sun_Glint_Angle',
            'Scan_Angle',
            'Observation',
        ]
        
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


def netcdf_to_parquet(input_file: str, output_path: str, date: str = None) -> None:
    """Convert NetCDF diagnostic file to Parquet format.
    
    Parameters
    ----------
    input_file : str
        Path to input NetCDF file
    output_path : str
        Path to output Parquet file or directory
    date : str, optional
        Date string for partitioning (YYYY-MM-DD format)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Read NetCDF data
    data = read_netcdf_diag(input_file)
    
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
    
    # Define observation variables to extract (same for all channels)
    # These will be sampled at every nchans-th value
    obs_variables = [
        ('observation_class', 'Observation_Class'),
        ('latitude', 'Latitude'),
        ('longitude', 'Longitude'),
        ('elevation', 'Elevation'),
        ('obs_time', 'Obs_Time'),
        ('scan_position', 'Scan_Position'),
        ('sat_zenith_angle', 'Sat_Zenith_Angle'),
        ('sat_azimuth_angle', 'Sat_Azimuth_Angle'),
        ('sol_zenith_angle', 'Sol_Zenith_Angle'),
        ('sol_azimuth_angle', 'Sol_Azimuth_Angle'),
        ('sun_glint_angle', 'Sun_Glint_Angle'),
        ('scan_angle', 'Scan_Angle'),
    ]
    
    # Add observation variables (same for all channels, so take every nchans-th value)
    # Assuming data is organized as: [obs1_ch1, obs1_ch2, ..., obs1_chN, obs2_ch1, obs2_ch2, ...]
    for output_name, netcdf_name in obs_variables:
        table_data[output_name] = pa.array(data[netcdf_name][::nchans])
    
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
        # Write as partitioned dataset
        os.makedirs(output_path, exist_ok=True)
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=['date'],
            existing_data_behavior='overwrite_or_ignore',
            basename_template='part-{i}.parquet'
        )
        print(f"Written partitioned parquet to: {output_path}")
    else:
        # Write as single file
        pq.write_table(table, output_path)
        print(f"Written parquet file to: {output_path}")


def process_netcdf_files(start_date: datetime,
                         end_date: datetime,
                         data_type: str,
                         input_pattern: str,
                         output_path: str,
                         partition_by_date: bool = True) -> None:
    """Process multiple NetCDF files and convert to Parquet.
    
    Parameters
    ----------
    start_date : datetime
        Start date for processing
    end_date : datetime
        End date for processing
    data_type : str
        Type of data (e.g., 'atms_n21')
    input_pattern : str
        Pattern for input files with {date} placeholder
        Example: '/path/to/diag_atms_n21_ges.{date}.nc'
    output_path : str
        Output directory for parquet files
    partition_by_date : bool
        Whether to partition output by date
    """
    day = timedelta(days=1)
    date = start_date
    
    while date <= end_date:
        date_str = date.strftime("%Y%m%d%H")
        date_partition = date.strftime("%Y-%m-%d")
        
        # Format input file path
        input_file = input_pattern.format(date=date_str)
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}, skipping...")
            date += day
            continue
        
        # Process file
        try:
            netcdf_to_parquet(
                input_file,
                output_path,
                date=date_partition if partition_by_date else None
            )
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
        
        date += day


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert NetCDF diagnostic files to Parquet format')
    parser.add_argument('input_file', help='Input NetCDF file path or pattern with {date} placeholder')
    parser.add_argument('output_path', help='Output Parquet file or directory path')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD) for batch processing')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD) for batch processing')
    parser.add_argument('--data-type', default='atms', help='Data type identifier')
    parser.add_argument('--partition', action='store_true', help='Partition output by date')
    parser.add_argument('--no-partition', dest='partition', action='store_false', help='Do not partition output')
    parser.set_defaults(partition=True)
    
    args = parser.parse_args()
    
    if args.start_date and args.end_date:
        # Batch processing mode
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        process_netcdf_files(
            start_date,
            end_date,
            args.data_type,
            args.input_file,
            args.output_path,
            partition_by_date=args.partition
        )
    else:
        # Single file mode
        date_str = None
        if args.partition:
            # Try to extract date from filename
            import re
            match = re.search(r'(\d{10})', args.input_file)
            if match:
                date_obj = datetime.strptime(match.group(1), "%Y%m%d%H")
                date_str = date_obj.strftime("%Y-%m-%d")
        
        netcdf_to_parquet(args.input_file, args.output_path, date=date_str)
