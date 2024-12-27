#!/bin/bash

#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/log-nemo-convert_%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/log-nemo-convert_%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=ckpt-cvs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

# Environment setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NEMO_LOG_MEMORY_USAGE=1
export WANDB_API_KEY=74bc2e3d0aa09e4d4e8a89659496aa4697714938
export NEMO_TESTING=1
export NEMO_REPO_DIR="/capstor/users/cscs/xyixuan/NeMo"

# Validate input
if [ "$#" -ne 1 ]; then
    echo "ERROR: Please provide checkpoint path"
    echo "Usage: sbatch $0 /path/to/checkpoint/megatron_checkpoint_file"
    exit 1
fi

# Get input path
CHECKPOINT_PATH="$1"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo "Working directory: $PWD"
echo "Processing checkpoint: $CHECKPOINT_PATH"

# Run the conversion script
# Using srun to ensure proper resource allocation
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    --unbuffered \
    /capstor/users/cscs/xyixuan/PDM/scripts/ckpt_conversion/convert.sh "$CHECKPOINT_PATH"

# Print job completion information
echo "Job completed at: $(date)"

# Usage:
# sbatch ./model-conversion-single.sh /users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/1.5B/llama_1.5B_Goldfish_K_54_H_13_GBS_120_EPOCH_76/results/checkpoints/megatron_llama_3_1_1.5B-step=3900-consumed_samples=468000.0
# sbatch ./model-conversion-single.sh /iopsstor/scratch/cscs/xyixuan/llama_1.5B_Goldfish_K_54_H_13_GBS_120_EPOCH_76/results/checkpoints/megatron_llama_3_1_1.5B-step=900-consumed_samples=108000.0
