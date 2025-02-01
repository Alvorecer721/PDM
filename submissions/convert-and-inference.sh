#!/bin/bash

# Validate input
if [ "$#" -ne 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory"
    exit 1
fi

EXPERIMENT_PATH="$1"
CKPTS_DIR="${EXPERIMENT_PATH}/results/checkpoints"

# Find latest checkpoint
LATEST_CKPT=$(ls -d ${CKPTS_DIR}/* | sort -t= -k2,2nr | head -n1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoints found in ${CKPTS_DIR}"
    exit 1
fi

echo "Using latest checkpoint: $LATEST_CKPT"

# Submit conversion job
CONVERT_JOB_ID=$(sbatch --parsable ./model-conversion-single.slurm "$LATEST_CKPT")
echo "Submitted checkpoint conversion job with ID: $CONVERT_JOB_ID"

# Submit inference job with dependency
INFERENCE_JOB_ID=$(sbatch --parsable --dependency=afterok:${CONVERT_JOB_ID} ./sparse-gutenberg-inference.slurm "$EXPERIMENT_PATH")
echo "Submitted inference job with ID: $INFERENCE_JOB_ID (will start after job ${CONVERT_JOB_ID} completes successfully)"

# Print status message
echo "Job chain submitted successfully!"
squeue -u $USER

# Usage:
# ./convert-and-inference.sh /iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_3968000
