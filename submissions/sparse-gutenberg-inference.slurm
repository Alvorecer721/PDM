#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/infer/hf-gfl-infer_%A_%a.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/infer/hf-gfl-infer_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-6

set -e  # Exit immediately if any command fails

# Setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

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
# declare -a offsets=(0 1 2 3 4 5 10 20 30 40 50 100)
# offset=${offsets[$SLURM_ARRAY_TASK_ID]}

# Define suffix length
declare -a suffix_lengths=(100 250 1000 1500 2000 2500 3000)
suffix_length=${suffix_lengths[$SLURM_ARRAY_TASK_ID]}

# Define prefix lengths
# declare -a prefix_lengths=(50 100 250 750 1000 1500 2000 3000 4000 5000)
# prefix_length=${prefix_lengths[$SLURM_ARRAY_TASK_ID]}

# Launch the distributed inference
srun torchrun \
    --nproc_per_node=4 \
    /capstor/users/cscs/xyixuan/PDM/src/infer/distributed_inference_sparse.py \
    --experiment-path "$EXPR_PATH" \
    --repetitions 1,2,3,4,8,16,24,32,48,64,96,128 \
    --gen-policy greedy \
    --offset 0 \
    --prefix-length 500 \
    --suffix-length "$suffix_length" \
    --batch-size 1

# Usage:
# sbatch ./sparse-gutenberg-inference.slurm /iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_1984000

# 64,128,256,512,1024,2048
# 1,2,3,4,8,16,24,32,48,64,96,128