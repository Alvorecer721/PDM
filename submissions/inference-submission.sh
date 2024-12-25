#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=72
#SBATCH --environment=/store/swissai/a06/.NeMo/container/nemo.toml
#SBATCH --error=/store/swissai/a06/.NeMo/Goldfish_Llama3/hf-gfl-infer_%j.err
#SBATCH --output=/store/swissai/a06/.NeMo/Goldfish_Llama3/hf-gfl-infer_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gfl-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=debug
# #SBATCH --time=06:00:00
# #SBATCH --account a06

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NEMO_LOG_MEMORY_USAGE=1
export WANDB_API_KEY=74bc2e3d0aa09e4d4e8a89659496aa4697714938
export NEMO_TESTING=1

# SLURM job to run the Megatron2HF conversion
srun \
    --output /store/swissai/a06/.NeMo/Goldfish_Llama3/hf-gfl-infer_%j.out \
    --error /store/swissai/a06/.NeMo/Goldfish_Llama3/hf-gfl-infer_%j.err \
    --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID \
    --wait 60 \
    --unbuffered \
    bash -c "cd /users/xyixuan/store/.NeMo/Goldfish_Llama3/PDM && ./scripts/todi_inference/run-inference.sh"