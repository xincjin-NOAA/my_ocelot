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


# Observation type from command line, default if not specified
OBS_TYPE=${1:-diag_surface_obs_uv}

# diag_urma_q, t, uv, cei, gst, hwv, ps, tca, vis, wst

python gen_data.py \
    2025-02-01 \
    2025-03-01 \
    ${OBS_TYPE} \
    diag_parquet

