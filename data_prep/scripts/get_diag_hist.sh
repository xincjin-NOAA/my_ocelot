#!/bin/sh -xvf

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <bdate> <edate>"
    echo "  bdate: Begin date in YYYYMMDDHH format (e.g., 2024010100)"
    echo "  edate: End date in YYYYMMDDHH format (e.g., 2024010318)"
    exit 1
fi

bdate=$1
edate=$2
cdate=${bdate}

expid=GDAS-ops
prod=v16.3

hpss_base_dir=/NCEPPROD/hpssprod/runhistory

#NDATE=/scratch2/NCEPDEV/nwprod/NCEPLIBS/utils/prod_util.v1.1.0/exec/ndate
NDATE=/home/Haixia.Liu/nwprod/util/exec/ndate
HPSSTAR=/home/Emily.Liu/bin/hpsstar

while [[ ${cdate} -le ${edate} ]]; do

   # for current analysis cycle
   data_dir=/scratch5/purged/Xin.C.Jin/my_ocelot/diag
   mkdir -p ${data_dir}
   cd ${data_dir}

   y4a=`echo $cdate | cut -c1-4`
   m2a=`echo $cdate | cut -c5-6`
   d2a=`echo $cdate | cut -c7-8`
   h2a=`echo $cdate | cut -c9-10`

   yyyymmdda=${y4a}${m2a}${d2a}
   hha=${h2a}
   yyyymmddhha=${yyyymmdda}${hha}

   # get required initial files from previous cycle 
   gdate=`$NDATE -6 ${cdate}`
   y4g=`echo $gdate | cut -c1-4`
   m2g=`echo $gdate | cut -c5-6`
   d2g=`echo $gdate | cut -c7-8`
   h2g=`echo $gdate | cut -c9-10`

   yyyymmddg=${y4g}${m2g}${d2g}
   hhg=${h2g}
   yyyymmddhhg=${yyyymmddg}${hhg}

   hpssa_dir=${hpss_base_dir}/rh${y4a}/${y4a}${m2a}/${yyyymmdda}
   hpssg_dir=${hpss_base_dir}/rh${y4g}/${y4g}${m2g}/${yyyymmddg}

   # =====================
   #  Get deterministics
   # =====================
   # Get initial conditions related to bias correction from previous (guess) cycle
   $HPSSTAR get ${hpssg_dir}/com_gfs_${prod}_gdas.${yyyymmddg}_${h2g}.gdas_restart.tar ./gdas.${yyyymmddg}/${hhg}/atmos/gdas.t${hhg}z.radstat
   cd ./gdas.${yyyymmddg}/${hhg}/atmos/
   # tar  -xvf  ./gdas.t${hh}z.radstat --wildcards --no-anchored  '*amsua*'
   tar  -xvf  ./gdas.t${hhg}z.radstat 
   rm ./gdas.t${hhg}z.radstat # tar -xvf gdas.t${h2g}z.radstat
   # extract the contents of the gzipped file and save to the output file
   find . -type f -name "*.gz" -exec gunzip {} +
   cdate=`$NDATE +6 ${cdate}`
   echo $cdate
done

