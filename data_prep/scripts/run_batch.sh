#!/bin/bash
#SBATCH -A da-cpu
#SBATCH -J gen_ocelot_data
#SBATCH -q batch
#SBATCH -t 08:00:00
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH -o jobs/gen_ocelot_data.%J.out
#SBATCH -e jobs/gen_ocelot_data.%J.err



source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/env.sh
source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/venv/bin/activate
export LOG_LEVEL=INFO
#srun -n 24 python gen_data.py 2024-01-01 2024-12-31 $1 zarr
#python gen_data.py 2024-01-01 2024-04-30 raw_radiosonde zarr
#python gen_data.py -p 2024-01-01 2024-12-31 atms cycle_parquet
#python gen_data.py -p 2024-01-01 2024-12-31 amsua cycle_parquet
#python gen_data.py -p 2024-01-01 2024-12-31 ssmis  cycle_parquet
#python gen_data.py -p 2024-01-01 2024-12-31 avhrr  cycle_parquet
#python gen_data.py 2024-01-01 2024-12-31 raw_surface_obs cycle_parquet
python gen_data.py 2024-01-01 2024-12-31 raw_radiosonde cycle_parquet
#python gen_data.py  -b  --slurm_account da-cpu 2024-01-01 2024-04-30 amsua cycle_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_atms diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_amsua diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_avhrr diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_ssmis diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_iasi diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_cris-fsr diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_gps diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_surface_obs_uv diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_surface_obs_t diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_sst diag_parquet
