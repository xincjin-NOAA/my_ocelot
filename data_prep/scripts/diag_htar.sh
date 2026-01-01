#!/bin/bash
set -euo pipefail

SRC_DIR="diag"
HPSS_DIR="/NCEPDEV/emc-da/2year/Xin.C.Jin/my_ocelot/diag"
MAX_JOBS=2   # HPSS-safe

mkdir -p logs

# Find months from gdas.YYYYMMDD
months=$(find "${SRC_DIR}" -maxdepth 1 -type d -name "gdas.20*" \
    | sed 's#.*/gdas\.##' \
    | grep -E '^[0-9]{8}$' \
    | cut -c1-6 | sort -u)

if [[ -z "${months}" ]]; then
    echo "ERROR: No gdas.YYYYMMDD directories found under ${SRC_DIR}"
    exit 1
fi

echo "Found months:"
echo "${months}"
echo "---------------------------"

htar_month() {
    month=$1
    echo "[START] ${month}"

    days=$(ls -d ${SRC_DIR}/gdas.${month}?? 2>/dev/null || true)
    if [[ -z "${days}" ]]; then
        echo "[SKIP] No days for ${month}"
        return
    fi

    echo "  Days: ${days}"

    htar -cvf "${HPSS_DIR}/diag_gdas_${month}.tar" ${days} \
        > logs/htar_${month}.log 2>&1

    echo "[DONE] ${month}"
}

export -f htar_month
export SRC_DIR HPSS_DIR

printf "%s\n" ${months} | \
    xargs -P ${MAX_JOBS} -I {} bash -c 'htar_month "$@"' _ {}
echo "All done"
