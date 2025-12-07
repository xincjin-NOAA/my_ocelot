#!/bin/bash

# Job submission script to process 2024 data in 12 monthly chunks
# Each month is submitted as a separate background job with nohup

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GET_DIAG_SCRIPT="${SCRIPT_DIR}/get_diag_hist.sh"
LOG_DIR="${SCRIPT_DIR}/../logs"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Define monthly date ranges for 2024
declare -a START_DATES=(
    "2024010100"  # January
    "2024020100"  # February
    "2024030100"  # March
    "2024040100"  # April
    "2024050100"  # May
    "2024060100"  # June
    "2024070100"  # July
    "2024080100"  # August
    "2024090100"  # September
    "2024100100"  # October
    "2024110100"  # November
    "2024120100"  # December
)

declare -a END_DATES=(
    "2024013118"  # January (31 days)
    "2024022918"  # February (29 days - leap year)
    "2024033118"  # March (31 days)
    "2024043018"  # April (30 days)
    "2024053118"  # May (31 days)
    "2024063018"  # June (30 days)
    "2024073118"  # July (31 days)
    "2024083118"  # August (31 days)
    "2024093018"  # September (30 days)
    "2024103118"  # October (31 days)
    "2024113018"  # November (30 days)
    "2024123118"  # December (31 days)
)

declare -a MONTH_NAMES=(
    "January"
    "February"
    "March"
    "April"
    "May"
    "June"
    "July"
    "August"
    "September"
    "October"
    "November"
    "December"
)

echo "=========================================="
echo "Submitting diagnostic data retrieval jobs"
echo "Year: 2024"
echo "Total jobs: 12 (one per month)"
echo "=========================================="
echo ""

# Submit jobs for each month
for i in {0..11}; do
    bdate=${START_DATES[$i]}
    edate=${END_DATES[$i]}
    month_name=${MONTH_NAMES[$i]}
    log_file="${LOG_DIR}/diag_${bdate}_${edate}.log"
    
    echo "Submitting job $((i+1))/12: ${month_name} 2024"
    echo "  Start date: ${bdate}"
    echo "  End date:   ${edate}"
    echo "  Log file:   ${log_file}"
    
    # Submit with nohup
    nohup ${GET_DIAG_SCRIPT} ${bdate} ${edate} > ${log_file} 2>&1 &
    
    # Get the process ID
    pid=$!
    echo "  Process ID: ${pid}"
    echo ""
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "To monitor jobs, use:"
echo "  ps aux | grep get_diag_hist.sh"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/diag_*.log"
echo ""
echo "To check progress for a specific month:"
echo "  tail -f ${LOG_DIR}/diag_2024<MM>0100_2024<MM><DD>18.log"
echo ""
