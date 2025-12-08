#!/bin/bash
#SBATCH -A da-cpu
#SBATCH -J gen_ocelot_data
#SBATCH -q batch
#SBATCH -t 04:00:00
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH -o gen_ocelot_data.%J.out
#SBATCH -e gen_ocelot_data.%J.err

# Usage: sbatch run_batch_netcdf.sh <obs_type>
# Example: sbatch run_batch_netcdf.sh atms

OBS_TYPE=${1:-atms}

START_DATE=2024-01-01
END_DATE=2024-12-31

echo "Processing NetCDF data:"
echo "  Start Date: $START_DATE"
echo "  End Date: $END_DATE"
echo "  Observation Type: $OBS_TYPE"

source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/env.sh
source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/venv/bin/activate
python gen_data.py $START_DATE $END_DATE $OBS_TYPE parquet --netcdf
