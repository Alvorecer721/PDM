#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/hf-gfl-infer_%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/hf-gfl-infer_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NEMO_LOG_MEMORY_USAGE=1
export NEMO_TESTING=1

# Get the master node hostname
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

# Launch the distributed training
srun python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    /capstor/users/cscs/xyixuan/PDM/src/infer/distributed_inference_sparse.py
