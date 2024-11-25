#!/bin/bash

# Set the last step
LAST_STEP=6225
STEP_INTERVAL=300
GLOBAL_BATCH_SIZE=120

# Calculate the number of files (rounding up to include the last step)
total_files=$(( (LAST_STEP + STEP_INTERVAL - 1) / STEP_INTERVAL ))

# Base paths
SOURCE_HOST="bristen"
DEST_HOST="rcp"
SOURCE_BASE="${SOURCE_HOST}:/capstor/store/cscs/swissai/a06/.NeMo/Goldfish_Llama3/1.5B/llama_1.5B_Goldfish_K_10_H_13_GBS_120_EPOCH_83/results/NeMo2HF"
DEST_BASE="${DEST_HOST}:/mnt/mlo/scratch/homes/yixuan/goldfish_ckpts/1b/k_10_h_13"

# Create destination directory first
DEST_PATH=$(echo "$DEST_BASE" | cut -d':' -f2)
ssh rcp "mkdir -p $DEST_PATH"

echo "Starting transfer of $total_files files..."

# Add start time
start_time=$(date +%s)
file_count=0

for step in $(seq $STEP_INTERVAL $STEP_INTERVAL $LAST_STEP); do
    file_count=$((file_count + 1))
    consumed=$((step * GLOBAL_BATCH_SIZE))
    filename="step=${step}-consumed=${consumed}.bin"
    
    # Calculate progress percentage
    progress=$((file_count * 100 / total_files))

    # Check if file exists at destination
    if ssh $DEST_HOST "test -f ${DEST_PATH}/${filename}"; then
        echo "[$file_count/$total_files - ${progress}%] ${filename} already exists, skipping..."
        continue
    fi
    
    # Calculate elapsed time and estimate remaining time
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $file_count -gt 1 ]; then
        eta=$((elapsed * (total_files - file_count + 1) / (file_count - 1)))
    else
        eta="calculating..."
    fi
    
    echo "[$file_count/$total_files - ${progress}%] Copying ${filename}... (elapsed: ${elapsed}s, ETA: ${eta}s)"
    
    # Remove pv, just use scp
    scp -3 "${SOURCE_BASE}/${filename}" "${DEST_BASE}/"
    
    sleep 1
done

# Handle the last step if it's not a multiple of STEP_INTERVAL
if (( LAST_STEP % STEP_INTERVAL != 0 )); then
    file_count=$((file_count + 1))
    consumed=$((LAST_STEP * GLOBAL_BATCH_SIZE))
    filename="step=${LAST_STEP}-consumed=${consumed}.bin"
    
    echo "[$file_count/$total_files - 100%] Copying final file ${filename}..."
    scp -3C "${SOURCE_BASE}/${filename}" "${DEST_BASE}/"
fi

echo "Transfer completed! Total time: $(($(date +%s) - start_time)) seconds"