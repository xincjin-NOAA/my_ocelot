#!/bin/bash
#SBATCH -A da-cpu
#SBATCH -J gen_ocelot_data
#SBATCH -q batch
#SBATCH -t 04:00:00
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH -o gen_ocelot_data.%J.out
#SBATCH -e gen_ocelot_data.%J.err



source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/env.sh
source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/venv/bin/activate
python gen_data.py 2024-01-01 2024-12-31 atms parquet --netcdf
