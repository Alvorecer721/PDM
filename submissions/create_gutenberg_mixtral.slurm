#!/bin/bash

# Parameters
#SBATCH --account=root
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/swap/swap_gutenberg_%j.err
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/swap/swap_gutenberg_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=swap_seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=debug
#SBATCH --exclude=nid006445

export CUDA_VISIBLE_DEVICES=0   # Only use the first GPU, ignoring GPUs 1,2,3

export TMPDIR=/capstor/scratch/cscs/xyixuan/tmp
export HF_HOME=/capstor/scratch/cscs/xyixuan/hf_cache
export HF_DATASETS_CACHE=/capstor/scratch/cscs/xyixuan/datasets_cache
export HUGGINGFACE_HUB_CACHE=/capstor/scratch/cscs/xyixuan/hub_cache

mkdir -p $TMPDIR
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HUGGINGFACE_HUB_CACHE

# Run the script with srun with appropriate flags
srun --cpu-bind=none \
    --wait 60 \
    --unbuffered \
    python /iopsstor/scratch/cscs/xyixuan/PDM/src/gutenberg/create_excerpt.py --output-dir /iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_en_8k_mixtral