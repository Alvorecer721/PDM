#!/bin/bash
# SLURM job for checkpoint conversion
#
# Path arguments:
#   experiment_path  - Output directory for converted models (torch/, HF/)
#   checkpoint_path  - Source torch-dist checkpoint (default: ${experiment_path}/checkpoints/3B)
#
# Example: sbatch /iopsstor/scratch/cscs/$USER/PDM/submissions/submit-convert.slurm  /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model2 \
#               --model-type apertus --checkpoint-path /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model2 --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer
#
# NOTE: The intermediate torch checkpoint will preserve the same tensor parallelism (TP)
#       as the source torch-dist checkpoint. The conversion script auto-detects TP from
#       the checkpoint and launches with the appropriate number of GPUs. Ensure this job
#       has enough GPUs allocated (default: 4) to match the checkpoint's TP setting.
#       The final HF checkpoint is always consolidated (TP=1).

#SBATCH --account=infra01
# Environment controlled via command-line: --environment=nemo_container
#SBATCH --gres=gpu:4
#SBATCH --job-name=ckpt-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/slurm/convert_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/slurm/convert_%j.err

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging
ulimit -c 0 # no core dumps

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Error: No experiment path provided!"
    echo "Usage: sbatch $0 <experiment_path> [--model-type TYPE] [--checkpoint-path PATH] [--tokenizer TOKENIZER] [--log-subdir SUBDIR]"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE        Model architecture type (default: llama3)"
    echo "                           Options: llama3, apertus"
    echo "  --checkpoint-path PATH   Direct path to torch distributed checkpoint"
    echo "                           (default: \${experiment_path}/checkpoints/3B)"
    echo "  --tokenizer TOKENIZER    HuggingFace tokenizer path or identifier"
    echo "  --log-subdir SUBDIR      Log subdirectory relative to PDM root"
    echo ""
    echo "Example:"
    echo "  sbatch $0 /path/to/megatron/experiment --model-type apertus"
    echo "  sbatch $0 /path/to/output --model-type apertus --checkpoint-path /path/to/checkpoint"
    echo ""
    echo "Note: This script only supports local Megatron checkpoints."
    echo "      HuggingFace model identifiers do not require conversion."
    exit 1
fi

EXPR_PATH="$1"
shift

# Default log subdirectory
LOG_SUBDIR="log/convert"

# Parse optional arguments and build conversion arguments
CONVERT_ARGS="$EXPR_PATH"
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-subdir)
            LOG_SUBDIR="$2"
            shift 2
            ;;
        --model-type|--checkpoint-path|--tokenizer)
            CONVERT_ARGS="$CONVERT_ARGS $1 $2"
            shift 2
            ;;
        *)
            CONVERT_ARGS="$CONVERT_ARGS $1"
            shift
            ;;
    esac
done

# Setup log directory and redirect output
LOG_DIR="/iopsstor/scratch/cscs/$USER/PDM/${LOG_SUBDIR}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/convert_${SLURM_JOB_ID}.out" 2>&1

# Print configuration
echo "=========================================================="
echo "submit-convert.slurm - Model Conversion Job Configuration"
echo "=========================================================="
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Log Directory:        ${LOG_DIR}"
echo "Arguments:            ${CONVERT_ARGS}"
echo "=========================================================="
echo ""

# Execute the conversion script inside the compute node using srun
echo "Executing conversion script PDM/scripts/swissai_megatron/convert.sh on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert.sh $CONVERT_ARGS

echo "Conversion End"
