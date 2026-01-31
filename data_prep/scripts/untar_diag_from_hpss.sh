#!/bin/bash
# Extract (untar) diag archives from HPSS that were created by diag_htar.sh
# Usage:
#   ./untar_diag_from_hpss.sh [month1 month2 ...]
#   If no months given, extracts all months found in HPSS_DIR (via hsi ls).
#   Months are YYYYMM (e.g. 202401 202402).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPSS_DIR="/NCEPDEV/emc-da/2year/Xin.C.Jin/my_ocelot/diag"
DEST_DIR="${SCRIPT_DIR}/../diag"
LOG_DIR="${SCRIPT_DIR}/../logs"
MAX_JOBS=2

mkdir -p "${DEST_DIR}"
mkdir -p "${LOG_DIR}"

# Resolve DEST_DIR if relative
DEST_DIR="$(cd "${DEST_DIR}" && pwd)"

list_months_on_hpss() {
    # hsi outputs full paths to stderr; match diag_gdas_YYYYMM.tar (exclude .idx)
    hsi "ls -1 ${HPSS_DIR}/" 2>&1 | grep -E 'diag_gdas_[0-9]{6}\.tar$' | sed 's/.*diag_gdas_\([0-9]\{6\}\)\.tar$/\1/' | sort -u
}

htar_extract_month() {
    local month=$1
    local log="${LOG_DIR}/htar_xvf_${month}.log"
    echo "[START] ${month}"
    (
        cd "${DEST_DIR}"
        htar -xvf "${HPSS_DIR}/diag_gdas_${month}.tar" > "${log}" 2>&1
    )
    echo "[DONE] ${month}"
}

export -f htar_extract_month
export HPSS_DIR DEST_DIR LOG_DIR

if [[ $# -gt 0 ]]; then
    months=("$@")
    echo "Extracting ${#months[@]} month(s): ${months[*]}"
else
    echo "Listing archives in ${HPSS_DIR} ..."
    echo "DEBUG: All files in directory (raw hsi ls output):"
    echo "---"
    hsi "ls -1 ${HPSS_DIR}/" 2>&1 || true
    echo "---"
    mapfile -t months < <(list_months_on_hpss)
    if [[ ${#months[@]} -eq 0 ]]; then
        echo "ERROR: No diag_gdas_YYYYMM.tar archives found in ${HPSS_DIR}"
        echo ""
        echo "DEBUG: Raw output from hsi ls (all files in directory):"
        echo "---"
        hsi "ls -1 ${HPSS_DIR}/" 2>&1 || true
        echo "---"
        echo ""
        echo "DEBUG: Raw output (one line at a time, hex for non-printable):"
        hsi "ls -1 ${HPSS_DIR}/" 2>&1 | od -c | head -100
        echo "---"
        exit 1
    fi
    echo "Found ${#months[@]} month(s): ${months[*]}"
fi

echo "Destination: ${DEST_DIR}"
echo "Logs: ${LOG_DIR}/htar_xvf_*.log"
echo "Parallel jobs: ${MAX_JOBS}"
echo ""
echo "Files to extract (for checking):"
for m in "${months[@]}"; do
    echo "  ${HPSS_DIR}/diag_gdas_${m}.tar"
done
echo "---------------------------"

printf '%s\n' "${months[@]}" | xargs -P "${MAX_JOBS}" -I {} bash -c 'htar_extract_month "$@"' _ {}

echo "All done."
