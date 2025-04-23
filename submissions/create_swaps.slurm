#!/bin/bash

# Parameters
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/swap/swap_gutenberg_%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/swap/swap_gutenberg_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=swap_seq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --time=01:30:00
##SBATCH --partition=debug

#SBATCH --time=03:00:00
#SBATCH --partition=normal

export CUDA_VISIBLE_DEVICES=0   # Only use the first GPU, ignoring GPUs 1,2,3

# Run the script with srun with appropriate flags
srun --cpu-bind=none \
    --wait 60 \
    --unbuffered \
    python /capstor/users/cscs/xyixuan/PDM/src/gutenberg/create_replicas_with_swapping.py --attempts-per-sequence 3