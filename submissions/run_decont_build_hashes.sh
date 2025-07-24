#!/bin/bash
#SBATCH --environment=ngc24-11
#SBATCH --job-name=decontam
#SBATCH --account=root
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=normal
#SBATCH --array=0-9
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/decontam_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/decontam_%A_%a.err


# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/PDM:$PYTHONPATH

# Activate conda environment
source /users/xyixuan/miniconda3_x86/etc/profile.d/conda.sh
conda activate decont

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export CTYPES_LIB_PATH=$CONDA_PREFIX/lib

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Change to project directory
cd /iopsstor/scratch/cscs/xyixuan/PDM

# Map array task ID to chunk number (00-10)
CHUNK_NUM=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
BASE_PATH="/capstor/store/cscs/swissai/infra01/users/xyixuan/finewebedu-sample-100BT"
FILE_PATH="${BASE_PATH}/finewebedu_000001_chunk_${CHUNK_NUM}.jsonl.gz"

# Run the decontamination analysis
echo "Starting contamination analysis for chunk ${CHUNK_NUM}..."
echo "Processing file: ${FILE_PATH}"
python /iopsstor/scratch/cscs/xyixuan/PDM/src/decont/build_index_hashes.py "${FILE_PATH}"
EXIT_CODE=$?

# Print completion info
echo "Job finished at: $(date)"

exit $EXIT_CODE