#!/bin/bash
#SBATCH --job-name=benchmark-summary
#SBATCH --account=infra01
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/slurm/summary_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/slurm/summary_%j.err

# This script generates a summary report for a benchmark run
# Usage: sbatch submit-job-summary.slurm <log_subdir> [--convert JOB_ID] [--lm-eval JOB_IDS...] [--mllm-eval JOB_IDS...]

ulimit -c 0 # no core dumps

# Use SLURM_SUBMIT_DIR to get the directory from which sbatch was called
# This is the submissions directory
SUBMISSIONS_DIR="${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(cd "$SUBMISSIONS_DIR/.." && pwd)"

# The log subdir will be used for output files
LOG_SUBDIR="$1"
shift

# Set output files based on log subdirectory
if [[ -n "$LOG_SUBDIR" ]]; then
    LOG_DIR="$REPO_ROOT/$LOG_SUBDIR"
    mkdir -p "$LOG_DIR"

    # Redirect output to log directory
    exec > "$LOG_DIR/summary_${SLURM_JOB_ID}.out" 2>&1
fi

echo "=========================================="
echo "Generating benchmark run summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Submissions Dir: $SUBMISSIONS_DIR"
echo "Log Directory: $LOG_SUBDIR"
echo "Full log path: $LOG_DIR"
echo "Repository Root: $REPO_ROOT"
echo "Started: $(date)"
echo ""

# Run the summary generation script
bash "$REPO_ROOT/scripts/generate_job_summary.sh" "$LOG_SUBDIR" "$@"

exit_code=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: $exit_code"

exit $exit_code