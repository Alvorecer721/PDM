#!/bin/bash

# Parameters
#SBATCH --account=infra01
#SBATCH --cpus-per-task=72
#SBATCH --environment=data-pipeline
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/preprocess/tokenise_%A_%a.err
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/preprocess/tokenise_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=tokenise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=normal

# Environment setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NEMO_LOG_MEMORY_USAGE=1
export WANDB_API_KEY=74bc2e3d0aa09e4d4e8a89659496aa4697714938
export NEMO_TESTING=1

# declare -a reps=(128 256 512 1024)
declare -a reps=(1 2 3 4 8 16 24 32 48 64 96 128)
rep=${reps[$SLURM_ARRAY_TASK_ID]}

# SLURM job to run the Megatron2HF conversion
# srun --cpu-bind=none \
#     --output /capstor/users/cscs/xyixuan/PDM/log/megatron_preprocess_%A_%a.out \
#     --error /capstor/users/cscs/xyixuan/PDM/log/megatron_preprocess_%A_%a.err \
#     --wait 60 \
#     --unbuffered \
#     bash -c "python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder /iopsstor/scratch/cscs/xyixuan/dataset/sparse_gutenberg/rep_${rep} --n-tasks 1 jsonl --dataset /iopsstor/scratch/cscs/xyixuan/dataset/gutenberg/rep_${rep}_text.jsonl"

# USAGE:
# sbatch --array=0-< length of reps - 1 > submissions/megatron-tokenisation.slurm
# sbatch --array=0-11 submissions/megatron-tokenisation.slurm

# export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/src:$PYTHONPATH
#
# srun --cpu-bind=none \
#     --wait 60 \
#     --unbuffered \
#     bash -c "python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder /iopsstor/scratch/cscs/xyixuan/dataset/sparse_gutenberg/rep_${rep} --n-tasks 1 jsonl --dataset /iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_continue/rep_${rep}_text.jsonl"

# NEW: OpenMathInstruct-2 tokenization
# USAGE: sbatch submissions/megatron-tokenisation.slurm

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/src:$PYTHONPATH

srun --cpu-bind=none \
    --wait 60 \
    --unbuffered \
    bash -c "python3 /iopsstor/scratch/cscs/xyixuan/data-pipeline-pretrain/examples/tokenize_megatron/preprocess_megatron.py \
        --tokenizer-name-or-path /capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer \
        --output-folder /capstor/store/cscs/swissai/infra01/vision-datasets/text_openorca_tokenized \
        --n-tasks 1 \
        --no-add-special-tokens \
        hf \
        --dataset parquet \
        --data-files /capstor/store/cscs/swissai/infra01/vision-datasets/text_openorca_formatted.parquet \
        --column text"
