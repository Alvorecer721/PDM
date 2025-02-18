#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=72
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/infer/hf-gfl-infer_%A_%a.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/infer/hf-gfl-infer_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=07:00:00
#SBATCH --array=0-9

set -e  # Exit immediately if any command fails

# Setup
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

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

# Define offsets
# declare -a offsets=(0 50 100)
# offset=${offsets[$SLURM_ARRAY_TASK_ID]}

# Define suffix length
# declare -a suffix_lengths=(50 100 250 500 1000 1500 2000 2500)
# suffix_length=${suffix_lengths[$SLURM_ARRAY_TASK_ID]}

# Define prefix lengths
declare -a prefix_lengths=(50 100 250 750 1000 1500 2000 3000 4000 5000)
prefix_length=${prefix_lengths[$SLURM_ARRAY_TASK_ID]}

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

if [ -n "${suffix_lengths+x}" ]; then
    verify_array_length "suffix_lengths" "${#suffix_lengths[@]}"
    suffix_length=${suffix_lengths[$SLURM_ARRAY_TASK_ID]}
fi

if [ -n "${prefix_lengths+x}" ]; then
    verify_array_length "prefix_lengths" "${#prefix_lengths[@]}"
    prefix_length=${prefix_lengths[$SLURM_ARRAY_TASK_ID]}
fi

# Reset to original state
set -e

srun pip install --user datasets

# Create the command string
CMD="srun torchrun \
    --nproc_per_node=4 \
    /capstor/users/cscs/xyixuan/PDM/src/infer/distributed_inference_swissai.py \
    --experiment-path \"$EXPR_PATH\" \
    --repetitions 1,2,4,8,16,24,32,48,64,96,128 \
    --gen-policy greedy \
    --offset 0 \
    --prefix-length $prefix_length \
    --suffix-length 500 \
    --batch-size 100"

# Echo the command to stdout only
echo "Executing command:" >&1
echo "$CMD" >&1

# Execute the command
eval "$CMD"

# Usage:
# sbatch ./swissai-convert-and-inference.slurm /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/Goldfish/llama3-1b-15n-8192sl-60gbsz-standard-no-bos

# 64,128,256,512,1024,2048
# 1,2,3,4,8,16,24,32,48,64,96,128