#!/bin/bash
#SBATCH -A da-cpu
#SBATCH -J gen_ocelot_data
#SBATCH -q batch
#SBATCH -t 08:00:00
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH -o jobs/gen_ocelot_data.%J.out
#SBATCH -e jobs/gen_ocelot_data.%J.err



source /scratch3/NCEPDEV/da/Xin.C.Jin/git/my_ocelot/data_prep/scripts/env.sh
export LOG_LEVEL=INFO

#python gen_data.py 2026-07-01 2026-07-07 diag_abi  diag_parquet
#python gen_data.py 2026-07-01 2026-07-07 diag_atms  diag_parquet
#python gen_data.py 2026-07-01 2026-07-07 diag_amsua  diag_parquet
#python gen_data.py 2026-07-01 2026-07-07 diag_cris-fsr  diag_parquet
#python gen_data.py 2026-07-01 2026-07-07 diag_iasi  diag_parquet
#python gen_data.py 2026-07-01 2026-07-07 diag_surface_obs_uv  diag_parquet
#python gen_data.py 2026-07-01 2026-07-07 diag_surface_obs_t  diag_parquet
python gen_data.py 2026-07-01 2026-07-07 diag_sst  diag_parquet


