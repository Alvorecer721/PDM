#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/eval/lm-eval_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/eval/lm-eval_%A.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=cp-conversion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging

# Setup
EXPR_PATH="$1"  # Ensure EXPR_PATH is provided
if [ -z "$EXPR_PATH" ]; then
    echo "Error: No experiment path provided!"
    echo "Usage: sbatch $0 /path/to/experiment/directory"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/$USER/PDM/log/eval

# Execute the script inside the compute node using srun
echo "Executing conversion and evaluation script on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert.sh "$EXPR_PATH"

echo "Job completed"