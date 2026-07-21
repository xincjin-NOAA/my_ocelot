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
#python gen_data.py 2024-01-01 2024-12-31 diag_surface_obs_uv diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_surface_obs_t diag_parquet
#python gen_data.py 2024-01-01 2024-12-31 diag_sst diag_parquet
# python gen_data.py 2025-02-01 2025-03-01 diag_urma_t diag_parquet
#python gen_data.py 2025-02-01 2025-03-01 diag_urma_q diag_parquet
#python gen_data.py 2025-02-01 2025-03-01 diag_urma_uv diag_parquet
#python gen_data.py 2025-02-01 2025-03-01 diag_urma_ps diag_parquet
python gen_data.py 2025-02-01 2025-03-01 diag_urma_cei diag_parquet
#python gen_data.py 2025-02-01 2025-02-02 anal_urma  diag_parquet


