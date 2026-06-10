 # Uncomment the following line if you don't like systemctl's auto-paging feature:
 # export SYSTEMD_PAGER=

# Prevent Python from loading ~/.local packages
# export PYTHONNOUSERSITE=1


module purge
#module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-gcc-11.4.1/install/modulefiles/Corei  # orig, but stack deleted.
module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-gcc-12.4.0/install/modulefiles/Core

module load stack-gcc
module load stack-openmpi
module load stack-python

#module load miniconda  # commented out bc of the new spac stack being used
module load ecflow
module load cmake
module load jedi-cmake
module load ecbuild
#module load odc
module load bufr
module load hdf5
module load netcdf-c
module load netcdf-cxx4
module load netcdf-fortran
module load udunits
module load eigen
module load boost
module load gsl-lite
module load eckit
module load fckit
module load atlas
module load py-pybind11
module load py-pip
module load json
module load json-schema-validator
module load python
module load nccmp
module load py-netcdf4
module load py-pyyaml
module load py-wxflow
module load py-pycodestyle
module load py-torch
module load cuda

export VALIDATE_PARAMETERS=1  # Make IODA do strict validation of YAML files

export PATH=/scratch3/NCEPDEV/da/Nicholas.Esposito/ocelot/installs/bin:$PATH
export LD_LIBRARY_PATH=/scratch3/NCEPDEV/da/Nicholas.Esposito/ocelot/installs/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/scratch3/NCEPDEV/da/Nicholas.Esposito/ocelot/installs/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch3/NCEPDEV/da/Nicholas.Esposito/ocelot/installs/lib64/python3.11/site-packages/:$PYTHONPATH
export PYTHONPATH=/scratch3/NCEPDEV/da/Nicholas.Esposito/ocelot/installs/lib64/python3.11/:$PYTHONPATH

source /scratch3/NCEPDEV/da/Xin.C.Jin/my_projects/ocelot/data/venv/bin/activate

# source /scratch3/NCEPDEV/da/Nicholas.Esposito/ocelot/ocelot_env/bin/activate

