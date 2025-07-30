#!/bin/bash
#SBATCH --account=root
#SBATCH --environment=ngc24-11
#SBATCH --job-name=decontam_opt
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/PDM/log/decont/decontam_opt_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/PDM/log/decont/decontam_opt_%A_%a.err
#SBATCH --array=0-16  # 17 jobs (for 169 chunks, 10 chunks each except last)

# Optimized version with more chunks per node since we have 288 CPUs

# Load environment based on architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # ARM architecture
    source /users/xyixuan/miniconda3/etc/profile.d/conda.sh
    conda activate decont
else
    # x86 architecture
    source /users/xyixuan/miniconda3_x86/etc/profile.d/conda.sh
    conda activate decont
fi

# Set working directory
cd /iopsstor/scratch/cscs/xyixuan/PDM

echo "Starting FineWeb index building - Array task $SLURM_ARRAY_TASK_ID"
echo "Time: $(date)"
echo "CPUs available: $(nproc)"

# Get list of all chunks
CHUNK_DIR="/iopsstor/scratch/cscs/xyixuan/dataset/finewebedu_chunks"
OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/fineweb"

# Create array of all chunk files, sorted for consistency
mapfile -t ALL_CHUNKS < <(find "$CHUNK_DIR" -type f -name "*_chunk_*" | sort)

# Calculate which chunks this array task should process (10 chunks per task)
CHUNKS_PER_TASK=10
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
    # Monitor system resources
    echo -e "\nSystem resources:"
    free -h
    echo -e "\nCPU info:"
    lscpu | grep -E "^CPU\(s\)|Thread\(s\)|Core\(s\)"
    
    # Process chunks using the parallel script
    # Use 10 workers to match chunks per task for optimal throughput
    echo -e "\nStarting parallel processing with 10 workers..."
    python src/decont/build_fineweb_indices_parallel.py \
        --workers 10 \
        --chunk-list "$CHUNK_LIST" \
        --output-dir "$OUTPUT_DIR" \
        --n-gram-size 13
        
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Successfully processed all chunks"
        
        # Quick validation - check that indices were created
        INDICES_CREATED=0
        while read -r chunk_path; do
            chunk_name=$(basename "$chunk_path" .jsonl.gz | sed 's/.jsonl$//')
            base_name=$(echo "$chunk_name" | cut -d'_' -f1-2)
            chunk_suffix=$(echo "$chunk_name" | cut -d'_' -f3-)
            
            if [ -f "$OUTPUT_DIR/$base_name/$chunk_suffix/input.index.hashes" ]; then
                ((INDICES_CREATED++))
            fi
        done < "$CHUNK_LIST"
        
        echo "Indices created: $INDICES_CREATED/$CHUNK_COUNT"
    else
        echo "Error occurred during processing (exit code: $EXIT_CODE)"
    fi
fi

# Clean up
rm -f "$CHUNK_LIST"

echo -e "\nArray task $SLURM_ARRAY_TASK_ID completed at: $(date)"