#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/infer/dense_%A_%a.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/infer/dense_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-7

set -e  # Exit immediately if any command fails

# Setup
EXPR_PATH="$1"

# Ensure EXPR_PATH is provided
if [ -z "$EXPR_PATH" ]; then
    echo "Error: No experiment path provided!"
    exit 1
fi

# Get the master node hostname
MASTER_PORT=$((10000 + ${SLURM_JOBID: -4}))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"

export MASTER_PORT
export MASTER_ADDR

# Define number of epoch
declare -a epochs=(1 2 4 8 16 32 64 80)
epoch=${epochs[$SLURM_ARRAY_TASK_ID]}


# Function to verify array length
verify_array_length() {
    local array_name=$1
    local array_size=$2
    
    if [ "$array_size" -ne "$SLURM_ARRAY_TASK_COUNT" ]; then
        echo "Error: Array size mismatch for $array_name!"
        echo "$array_name array has $array_size elements"
        echo "SLURM array size is $SLURM_ARRAY_TASK_COUNT (e.g. --array=0-7 means 8 tasks)"
        exit 1
    fi
}

# Check if arrays are defined using set -u to avoid undefined variable errors
set +u

# Verify arrays if they are uncommented
if [ -n "${offsets+x}" ]; then
    verify_array_length "offsets" "${#offsets[@]}"
    offset=${offsets[$SLURM_ARRAY_TASK_ID]}
fi


# Reset to original state
set -e

# Create the command string
CMD="srun torchrun \
    --nproc_per_node=4 \
    /capstor/users/cscs/xyixuan/PDM/src/infer/distributed_inference_dense.py \
    --experiment-path \"$EXPR_PATH\" \
    --gen-policy greedy \
    --num-epoch $epoch \
    --offset 50 \
    --prefix-length 500 \
    --suffix-length 500 \
    --batch-size 20 \
    --num-proc 100"

# Echo the command to stdout only
echo "Executing command:" >&1
echo "$CMD" >&1

# Execute the command
eval "$CMD"

# Usage:
# sbatch dense-gutenberg-inference.slurm /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz

# 64,128,256,512,1024,2048
# 1,2,3,4,8,16,24,32,48,64,96,128