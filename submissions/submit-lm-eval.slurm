#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=288
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/eval/lm-eval_%A.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/eval/lm-eval_%A.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=lm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00

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
mkdir -p /capstor/users/cscs/$USER/PDM/log/eval

# Execute the script inside the compute node using srun
echo "Executing conversion and evaluation script on compute node..."
srun /capstor/users/cscs/xyixuan/PDM/scripts/swissai_megatron/convert-and-lm-eval.sh "$EXPR_PATH"

echo "Job completed"