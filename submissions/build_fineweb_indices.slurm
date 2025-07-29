#!/bin/bash
#SBATCH --account=root
#SBATCH --environment=data-pipeline
#SBATCH --job-name=decontam
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/decontam_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/decontam_%A_%a.err
#SBATCH --array=0-6  # 7 jobs (for 169 chunks, 25 chunks each except last)


# Load environment based on architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    # Only activate conda on non-ARM architectures
    source /users/xyixuan/miniconda3/etc/profile.d/conda.sh
    conda activate decont
else
    echo "Running on ARM architecture ($ARCH), using SBATCH environment"
fi

# Set working directory
cd /iopsstor/scratch/cscs/xyixuan/PDM

echo "Starting FineWeb index building - Array task $SLURM_ARRAY_TASK_ID"
echo "Time: $(date)"

# Get list of all chunks
CHUNK_DIR="/iopsstor/scratch/cscs/xyixuan/dataset/finewebedu_chunks"
OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/fineweb"

# Create array of all chunk files, sorted for consistency
mapfile -t ALL_CHUNKS < <(find "$CHUNK_DIR" -type f -name "*_chunk_*" | sort)

# Calculate which chunks this array task should process (25 chunks per task)
CHUNKS_PER_TASK=25
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNKS_PER_TASK))
END_IDX=$((START_IDX + CHUNKS_PER_TASK - 1))

echo "Processing chunks $START_IDX to $END_IDX"
echo "Total chunks available: ${#ALL_CHUNKS[@]}"

# Create a temporary file with the list of chunks for this task
CHUNK_LIST=$(mktemp)
CHUNK_COUNT=0
for i in $(seq $START_IDX $END_IDX); do
    if [ $i -lt ${#ALL_CHUNKS[@]} ]; then
        echo "${ALL_CHUNKS[$i]}" >> "$CHUNK_LIST"
        ((CHUNK_COUNT++))
    fi
done

echo "This task will process $CHUNK_COUNT chunks"

if [ $CHUNK_COUNT -gt 0 ]; then
    # Process chunks using the parallel script
    # Use 25 workers to process 25 chunks in parallel (1 worker per chunk)
    python src/decont/build_fineweb_indices_parallel.py \
        --workers 25 \
        --chunk-list "$CHUNK_LIST" \
        --output-dir "$OUTPUT_DIR" \
        --n-gram-size 13
fi

# Clean up
rm -f "$CHUNK_LIST"

echo -e "\nArray task $SLURM_ARRAY_TASK_ID completed at: $(date)"