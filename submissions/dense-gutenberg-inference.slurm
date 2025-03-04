#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/infer/dense/dense_%A_%a.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/infer/dense/dense_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-71

set -e  # Exit immediately if any command fails

# Setup
EXPR_PATH="$1"
PILOT_MODE="$2"  # Add a pilot mode flag argument

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

# Find the last epoch by getting the highest iteration and dividing by 125
if [ -d "${EXPR_PATH}/HF" ]; then
    # Find the highest iteration number from the directory names
    HIGHEST_ITER=$(ls -d "${EXPR_PATH}/HF/iter_"* 2>/dev/null | sed 's/.*iter_//' | sort -n | tail -1)
    
    if [ -n "$HIGHEST_ITER" ]; then
        # Remove leading zeros and calculate last epoch by dividing by 125
        HIGHEST_ITER_CLEAN=$(echo "$HIGHEST_ITER" | sed 's/^0*//')
        LAST_EPOCH=$((HIGHEST_ITER_CLEAN / 125))
        echo "Found highest iteration: $HIGHEST_ITER ($HIGHEST_ITER_CLEAN without leading zeros), last epoch: $LAST_EPOCH"
    else
        echo "Warning: No iteration directories found at ${EXPR_PATH}/HF/"
        exit 1  
    fi
else
    echo "Warning: HF directory not found at ${EXPR_PATH}/HF, using default last epoch"
    exit 1 
fi

# Define arrays for epochs and offsets
# declare -a epochs=(1 2 4 8 16 32 64 82)
# declare -a offsets=(0 1 2 4 8 16 32 64 128 256)
declare -a epochs=($LAST_EPOCH)
declare -a offsets=(0)

# Check if pilot mode is enabled
if [ "$PILOT_MODE" = "pilot" ]; then
    echo "Running in PILOT MODE - using only first epoch and offset"
    # Use only the first elements, regardless of array task ID
    epoch=${epochs[0]}
    offset=${offsets[0]}
    echo "Fixed pilot values: epoch=$epoch, offset=$offset"
else
    # Calculate total combinations
    TOTAL_COMBINATIONS=$((${#epochs[@]} * ${#offsets[@]}))

    # Check if SLURM array size matches total combinations
    if [ "$SLURM_ARRAY_TASK_COUNT" -ne "$TOTAL_COMBINATIONS" ]; then
        echo "Warning: SLURM array size (${SLURM_ARRAY_TASK_COUNT}) does not match total combinations (${TOTAL_COMBINATIONS})"
        echo "Please adjust your --array parameter to --array=0-$((TOTAL_COMBINATIONS-1))"
    fi

    # Calculate which epoch and offset to use for this task
    EPOCH_INDEX=$((SLURM_ARRAY_TASK_ID / ${#offsets[@]}))
    OFFSET_INDEX=$((SLURM_ARRAY_TASK_ID % ${#offsets[@]}))

    # Get the actual epoch and offset values
    epoch=${epochs[$EPOCH_INDEX]}
    offset=${offsets[$OFFSET_INDEX]}
    
    echo "Running with epoch=$epoch, offset=$offset (Task ID: $SLURM_ARRAY_TASK_ID)"
fi

# Create the command string
CMD="srun torchrun \
    --nproc_per_node=4 \
    /capstor/users/cscs/xyixuan/PDM/src/infer/distributed_inference_dense.py \
    --experiment-path \"$EXPR_PATH\" \
    --gen-policy greedy \
    --num-epoch $epoch \
    --offset $offset \
    --prefix-length 500 \
    --suffix-length 500 \
    --batch-size 20 \
    --num-proc 100"

# Echo the command to stdout
echo "Executing command:" >&1
echo "$CMD" >&1

# Execute the command
eval "$CMD"

# Usage:
# Regular run:
# sbatch --array=0-71 dense-gutenberg-inference.slurm /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz
#
# Pilot run (only run epoch=1, offset=1):
# sbatch --array=0 dense-gutenberg-inference.slurm /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz pilot