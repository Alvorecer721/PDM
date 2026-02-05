#!/bin/bash

# Generate a summary report for a benchmark run
# Usage: generate_job_summary.sh <log_subdir> [--convert JOB_ID] [--lm-eval JOB_IDS...] [--mllm-eval JOB_IDS...]

set -euo pipefail

# Parse arguments
LOG_SUBDIR=""
CONVERT_JOB=""
LM_EVAL_JOBS=()
MLLM_EVAL_JOBS=()

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <log_subdir> [--convert JOB_ID] [--lm-eval JOB_IDS...] [--mllm-eval JOB_IDS...]"
    exit 1
fi

LOG_SUBDIR="$1"
shift

current_category=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --convert)
            current_category="convert"
            shift
            ;;
        --lm-eval)
            current_category="lm-eval"
            shift
            ;;
        --mllm-eval)
            current_category="mllm-eval"
            shift
            ;;
        *)
            if [[ "$current_category" == "convert" ]]; then
                CONVERT_JOB="$1"
            elif [[ "$current_category" == "lm-eval" ]]; then
                LM_EVAL_JOBS+=("$1")
            elif [[ "$current_category" == "mllm-eval" ]]; then
                MLLM_EVAL_JOBS+=("$1")
            fi
            shift
            ;;
    esac
done

# Resolve full path for log directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/$LOG_SUBDIR"

echo "--> config: generate_job_summary.sh <--"
echo "Script Dir: $SCRIPT_DIR"
echo "Repo Root:  $REPO_ROOT"
echo "Log Dir:    $LOG_DIR"
echo ""

# Wait a bit for jobs to be registered in accounting system
sleep 10

# Output file
SUMMARY_FILE="$LOG_DIR/summary.txt"

# Collect all job IDs
ALL_JOBS=()
[[ -n "$CONVERT_JOB" ]] && ALL_JOBS+=("$CONVERT_JOB")
ALL_JOBS+=("${LM_EVAL_JOBS[@]}")
ALL_JOBS+=("${MLLM_EVAL_JOBS[@]}")

# Query job information
declare -A JOB_INFO
declare -A JOB_TYPE
declare -A JOB_STATE
declare -A JOB_ELAPSED
declare -A JOB_START
declare -A JOB_END
declare -A JOB_EXIT_CODE
declare -A JOB_GROUP_NAME

# Initialize counters
TOTAL_JOBS=0
SUCCESSFUL_JOBS=0
FAILED_JOBS=0

# Track timing
EARLIEST_START=""
LATEST_END=""
TOTAL_ELAPSED_SECONDS=0

# Function to convert time to seconds
time_to_seconds() {
    local time_str="$1"
    local seconds=0

    # Handle format: [DD-[HH:]]MM:SS
    if [[ "$time_str" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        # DD-HH:MM:SS
        seconds=$((${BASH_REMATCH[1]} * 86400 + ${BASH_REMATCH[2]} * 3600 + ${BASH_REMATCH[3]} * 60 + ${BASH_REMATCH[4]}))
    elif [[ "$time_str" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        # HH:MM:SS
        seconds=$((${BASH_REMATCH[1]} * 3600 + ${BASH_REMATCH[2]} * 60 + ${BASH_REMATCH[3]}))
    elif [[ "$time_str" =~ ^([0-9]+):([0-9]+)$ ]]; then
        # MM:SS
        seconds=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
    fi

    echo "$seconds"
}

# Function to format seconds to human readable
seconds_to_human() {
    local total_seconds=$1
    local days=$((total_seconds / 86400))
    local hours=$(((total_seconds % 86400) / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    local output=""
    [[ $days -gt 0 ]] && output="${days}d "
    [[ $hours -gt 0 ]] && output="${output}${hours}h "
    [[ $minutes -gt 0 ]] && output="${output}${minutes}m "
    output="${output}${seconds}s"

    echo "$output"
}

# Function to compare timestamps (returns 1 if first is earlier, 0 otherwise)
is_earlier() {
    local time1="$1"
    local time2="$2"

    if [[ -z "$time2" ]]; then
        echo 1
        return
    fi

    if [[ "$time1" < "$time2" ]]; then
        echo 1
    else
        echo 0
    fi
}

# Function to compare timestamps (returns 1 if first is later, 0 otherwise)
is_later() {
    local time1="$1"
    local time2="$2"

    if [[ -z "$time2" ]]; then
        echo 1
        return
    fi

    if [[ "$time1" > "$time2" ]]; then
        echo 1
    else
        echo 0
    fi
}

# Query each job
for job_id in "${ALL_JOBS[@]}"; do
    # Determine job type
    if [[ -n "$CONVERT_JOB" && "$job_id" == "$CONVERT_JOB" ]]; then
        job_type="convert"
    elif [[ " ${LM_EVAL_JOBS[@]} " =~ " ${job_id} " ]]; then
        job_type="lm-eval"
    else
        job_type="mllm-eval"
    fi

    JOB_TYPE[$job_id]="$job_type"

    # Query sacct
    sacct_output=$(sacct -j "$job_id" --format=JobID,JobName,State,Start,End,Elapsed,ExitCode -P --noheader 2>/dev/null || echo "")

    if [[ -z "$sacct_output" ]]; then
        # Job not found in accounting system yet
        JOB_STATE[$job_id]="PENDING"
        continue
    fi

    # Parse first line (main job, not steps)
    job_line=$(echo "$sacct_output" | head -n 1)

    IFS='|' read -r job_id_full job_name state start_time end_time elapsed exit_code <<< "$job_line"

    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    JOB_INFO[$job_id]="$job_name"
    JOB_STATE[$job_id]="$state"
    JOB_ELAPSED[$job_id]="$elapsed"
    JOB_START[$job_id]="$start_time"
    JOB_END[$job_id]="$end_time"
    JOB_EXIT_CODE[$job_id]="$exit_code"

    # Extract group name from job name if present (format: lm-eval-<group> or mllm-eval-<group>)
    if [[ "$job_name" =~ (lm|mllm)-eval-(.+) ]]; then
        JOB_GROUP_NAME[$job_id]="${BASH_REMATCH[2]}"
    fi

    # Count successes/failures
    if [[ "$state" == "COMPLETED" ]]; then
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
    elif [[ "$state" != "PENDING" && "$state" != "RUNNING" ]]; then
        FAILED_JOBS=$((FAILED_JOBS + 1))
    fi

    # Track timing
    if [[ -n "$start_time" && "$start_time" != "Unknown" ]]; then
        if [[ $(is_earlier "$start_time" "$EARLIEST_START") -eq 1 ]]; then
            EARLIEST_START="$start_time"
        fi
    fi

    if [[ -n "$end_time" && "$end_time" != "Unknown" ]]; then
        if [[ $(is_later "$end_time" "$LATEST_END") -eq 1 ]]; then
            LATEST_END="$end_time"
        fi
    fi

    # Add to total elapsed time
    if [[ -n "$elapsed" ]]; then
        elapsed_seconds=$(time_to_seconds "$elapsed")
        TOTAL_ELAPSED_SECONDS=$((TOTAL_ELAPSED_SECONDS + elapsed_seconds))
    fi
done

# Calculate time span
TIME_SPAN_SECONDS=0
if [[ -n "$EARLIEST_START" && -n "$LATEST_END" ]]; then
    # Convert to epoch seconds and calculate difference
    start_epoch=$(date -d "$EARLIEST_START" +%s 2>/dev/null || echo "0")
    end_epoch=$(date -d "$LATEST_END" +%s 2>/dev/null || echo "0")
    TIME_SPAN_SECONDS=$((end_epoch - start_epoch))
fi

# Generate summary report
{
    echo "========================================"
    echo "Benchmark Run Summary"
    echo "========================================"
    echo "Log Directory: $LOG_SUBDIR"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Job Statistics:"
    echo "  Total Jobs: $TOTAL_JOBS"
    echo "  Successful: $SUCCESSFUL_JOBS"
    echo "  Failed: $FAILED_JOBS"
    echo ""

    # List failed jobs
    if [[ $FAILED_JOBS -gt 0 ]]; then
        echo "Failed Jobs:"
        for job_id in "${ALL_JOBS[@]}"; do
            state="${JOB_STATE[$job_id]:-UNKNOWN}"
            if [[ "$state" != "COMPLETED" && "$state" != "PENDING" && "$state" != "RUNNING" ]]; then
                job_type="${JOB_TYPE[$job_id]}"
                job_name="${JOB_INFO[$job_id]:-unknown}"
                exit_code="${JOB_EXIT_CODE[$job_id]:-unknown}"
                group_name="${JOB_GROUP_NAME[$job_id]:-}"

                if [[ -n "$group_name" ]]; then
                    echo "  [$job_type] Job $job_id (group: $group_name) - $state (Exit code: $exit_code)"
                else
                    echo "  [$job_type] Job $job_id - $state (Exit code: $exit_code)"
                fi
            fi
        done
        echo ""
    fi

    # Timing information
    echo "Job Timing:"
    if [[ -n "$EARLIEST_START" ]]; then
        echo "  First job started: $EARLIEST_START"
    fi
    if [[ -n "$LATEST_END" ]]; then
        echo "  Last job ended: $LATEST_END"
    fi
    if [[ $TIME_SPAN_SECONDS -gt 0 ]]; then
        echo "  Total time span: $(seconds_to_human $TIME_SPAN_SECONDS)"
    fi
    if [[ $TOTAL_ELAPSED_SECONDS -gt 0 ]]; then
        echo "  Combined runtime: $(seconds_to_human $TOTAL_ELAPSED_SECONDS)"
    fi
    echo ""

    # Detailed job list
    echo "Detailed Job List:"
    for job_id in "${ALL_JOBS[@]}"; do
        job_type="${JOB_TYPE[$job_id]}"
        job_name="${JOB_INFO[$job_id]:-unknown}"
        state="${JOB_STATE[$job_id]:-UNKNOWN}"
        elapsed="${JOB_ELAPSED[$job_id]:-0:00}"
        group_name="${JOB_GROUP_NAME[$job_id]:-}"

        if [[ -n "$group_name" ]]; then
            echo "  [$job_type] Job $job_id ($group_name) - $state ($elapsed)"
        else
            echo "  [$job_type] Job $job_id - $state ($elapsed)"
        fi
    done
    echo "========================================"
} > "$SUMMARY_FILE"

echo "Summary report generated: $SUMMARY_FILE"
cat "$SUMMARY_FILE"