#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=72
#SBATCH --environment=data-pipeline
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/megatron_preprocess_%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/megatron_preprocess_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=tokenise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:59
#SBATCH --partition=normal
# #SBATCH --time=06:00:00
# #SBATCH --account a06

# Environment setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NEMO_LOG_MEMORY_USAGE=1
export WANDB_API_KEY=74bc2e3d0aa09e4d4e8a89659496aa4697714938
export NEMO_TESTING=1

# SLURM job to run the Megatron2HF conversion
srun \
    --output /capstor/users/cscs/xyixuan/PDM/log/megatron_preprocess_%j.out \
    --error /capstor/users/cscs/xyixuan/PDM/log/megatron_preprocess_%j.err \
    --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID \
    --wait 60 \
    --unbuffered \
    bash -c "python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder /capstor/scratch/cscs/xyixuan/gutenberg --n-tasks 16 jsonl --dataset /capstor/users/cscs/xyixuan/data/raw/gutenberg_en_8k"