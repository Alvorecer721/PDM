#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/convert/convert_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/convert/convert_%A.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=ckpt-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Error: No experiment path provided!"
    echo "Usage: sbatch $0 <experiment_path>"
    echo ""
    echo "Example:"
    echo "  sbatch $0 /path/to/megatron/experiment"
    echo ""
    echo "Note: This script only supports local Megatron checkpoints."
    echo "      HuggingFace model identifiers do not require conversion."
    exit 1
fi

EXPR_PATH="$1"

# Print configuration
echo "========================================"
echo "Model Conversion Job Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "========================================"
echo ""

# Create log directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/$USER/PDM/log/convert

# Execute the conversion script inside the compute node using srun
echo "Executing conversion script on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert.sh "$EXPR_PATH"

echo "Conversion completed"
