#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --job-name=mauve
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/mauve/mauve_%A_%a.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/mauve/mauve_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=0

# Load any necessary modules based on your cluster configuration
# module load ...

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mauve_env

# Define the array of offsets
OFFSETS=(512) # 1 2 4 8 16 32 64 128 256 1024 2048
CURRENT_OFFSET=${OFFSETS[$SLURM_ARRAY_TASK_ID]}

# Run the script
python /capstor/users/cscs/xyixuan/PDM/src/verbatim_eval/compute_mauve.py \
  --expr "llama3-8b-15n-8192sl-60gbsz-standard" \
  --offsets $CURRENT_OFFSET \
  --prefix-lengths 500 \
  --suffix-lengths 500 \
  --repetitions 128 \
  --mode SparseGutenberg


# --mode DenseGutenberg
