#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/hf-gfl-infer_%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/hf-gfl-infer_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

EXPR_PATH="$1"

# Get the master node hostname
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

export MASTER_PORT
export MASTER_ADDR

# Launch the distributed training
srun torchrun \
    --nproc_per_node=4 \
    /capstor/users/cscs/xyixuan/PDM/src/infer/distributed_inference_sparse.py --experiment-path $EXPR_PATH --repetitions 64,128,256,512,1024,2048 --gen-policy greedy

# Usage:
# sbatch ./sparse-gutenberg-inference.slurm /iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_1984000