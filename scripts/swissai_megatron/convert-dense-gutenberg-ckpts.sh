#!/bin/bash

set -e

# Configuration
ITERATIONS_PER_EPOCH=125
DRY_RUN=false

# Function to log messages with timestamp
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if an iteration is already converted
is_already_converted() {
  local iteration="$1"
  local expr_dir="$2"
  
  # If in dry run mode, always return false (not converted)
  if $DRY_RUN; then
    return 1
  fi
  
  # Check for the existence of the converted checkpoint directory or files
  local iter_dir="${expr_dir}/iter_${iteration}"
  
  if [ -d "$iter_dir" ] && [ -f "${iter_dir}/model.safetensors" ]; then
    return 0  # true, already converted
  fi
  
  return 1  # false, not yet converted
}

# Function to run the conversion command
run_conversion() {
  local checkpoint_dir="$1"
  local expr_dir="$2"
  
  log_message "Running conversion command..."
  
  # If in dry run mode, skip the actual conversion
  if $DRY_RUN; then
    log_message "[DRY RUN] Would execute conversion command here"
    return 0
  fi
  
  CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun /capstor/users/cscs/xyixuan/PDM/src/convert/convert_torch_dist_to_torch.py \
    --bf16 \
    --use-precision-aware-optimizer \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --wgrad-deferral-limit 50 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --main-grads-dtype bf16 \
    --attention-dropout 0.0 \
    --use-rope-scaling \
    --swiglu \
    --group-query-attention \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model nvidia/OpenMath2-Llama3.1-8B \
    --make-vocab-size-divisible-by 128 \
    --num-layers 16 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --max-position-embeddings 8192 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rope-scaling-factor 32 \
    --normalization RMSNorm \
    --seq-length 8192 \
    --load "${checkpoint_dir}" \
    --ckpt-convert-save "${expr_dir}"
    
  return $?
}

# Function to update the iteration file
update_iteration_file() {
  local file="$1"
  local iteration="$2"
  
  if $DRY_RUN; then
    log_message "[DRY RUN] Would update $file with iteration $iteration (epoch $((iteration / ITERATIONS_PER_EPOCH)))"
    return 0
  fi
  
  echo "$iteration" > "$file"
  log_message "Updated $file with iteration $iteration (epoch $((iteration / ITERATIONS_PER_EPOCH)))"
}

# Function to generate powers of 2 up to a limit
generate_powers_of_two() {
  local limit="$1"
  local powers=()
  local power=0
  
  while true; do
    local current_power=$(echo "2^$power" | bc)
    if [ "$current_power" -gt "$limit" ]; then
      break
    fi
    powers+=($current_power)
    power=$((power + 1))
  done
  
  echo "${powers[@]}"
}

# Function to print array elements with a specific format
# Usage: print_array_formatted [array] [format]
print_array_formatted() {
  local -n array=$1
  local format=$2
  local result=""
  
  for item in "${array[@]}"; do
    if [ -n "$result" ]; then
      result="$result, "
    fi
    result="$result$(printf "$format" "$item")"
  done
  
  echo "$result"
}

# Function to process a single iteration
process_iteration() {
  local iteration="$1"
  local iteration_file="$2"
  local checkpoint_dir="$3"
  local expr_dir="$4"
  
  local epoch=$((iteration / ITERATIONS_PER_EPOCH))
  
  log_message "=============================================================================="
  if $DRY_RUN; then
    log_message "[DRY RUN] Processing checkpoint at iteration $iteration (epoch $epoch)"
  else
    log_message "Processing checkpoint at iteration $iteration (epoch $epoch)"
  fi
  
  # Check if already converted
  if is_already_converted "$iteration" "$expr_dir"; then
    log_message "Checkpoint at epoch $epoch (iteration $iteration) already converted. Skipping."
    log_message "=============================================================================="
    log_message ""
    return 0
  fi
  
  # Update the iteration file
  update_iteration_file "$iteration_file" "$iteration"
  
  # Wait a moment for file system to sync
  if ! $DRY_RUN; then
    sleep 1
  fi
  
  # Run the conversion command
  run_conversion "$checkpoint_dir" "$expr_dir"
  
  if [ $? -ne 0 ]; then
    log_message "Conversion failed for checkpoint at epoch $epoch (iteration $iteration)."
    log_message "=============================================================================="
    log_message ""
    return 1
  else
    if $DRY_RUN; then
      log_message "[DRY RUN] Conversion would complete successfully for checkpoint at epoch $epoch (iteration $iteration)."
    else
      log_message "Conversion completed successfully for checkpoint at epoch $epoch (iteration $iteration)."
    fi
  fi
  
  if $DRY_RUN; then
    log_message "[DRY RUN] Completed processing for epoch $epoch (iteration $iteration)"
  else
    log_message "Completed processing for epoch $epoch (iteration $iteration)"
  fi
  log_message "=============================================================================="
  log_message ""
  
  return 0
}

# Function to print usage
print_usage() {
  echo "Usage: $0 [OPTIONS] <expr_dir>"
  echo "Example: $0 --dry-run /path/to/DenseGutenberg/llama3-1b-standard-80gbsz"
  echo ""
  echo "Options:"
  echo "  --dry-run, -n    Run in dry-run mode without making any actual changes"
  echo "  --help, -h       Display this help message"
}

# Main function
main() {
  # Parse command line arguments
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --dry-run|-n)
        DRY_RUN=true
        shift
        ;;
      --help|-h)
        print_usage
        return 0
        ;;
      *)
        break
        ;;
    esac
  done

  # Check arguments
  if [ $# -lt 1 ]; then
    print_usage
    return 1
  fi

  # Set experiment directory
  local expr_dir="$1"
  local checkpoint_dir="${expr_dir}/checkpoints"
  local iteration_file="${checkpoint_dir}/latest_checkpointed_iteration.txt"

  if $DRY_RUN; then
    log_message "[DRY RUN] Running in dry-run mode - no actual changes will be made"
  fi

  # Check if iteration file exists
  if [ ! -f "$iteration_file" ]; then
    log_message "Error: Checkpoint iteration file not found: $iteration_file"
    return 1
  fi

  # Read the last checkpoint iteration
  local last_iteration=$(cat "$iteration_file")
  local last_epoch=$((last_iteration / ITERATIONS_PER_EPOCH))
  log_message "Last checkpoint: iteration $last_iteration (epoch $last_epoch)"

  # Divide by iterations per epoch to get epoch count
  local epochs=$(echo "$last_iteration / $ITERATIONS_PER_EPOCH" | bc)
  log_message "Total epochs: $epochs"

  # Generate powers of 2 list (in terms of epochs)
  local powers_list=($(generate_powers_of_two "$epochs"))
  
  # Calculate iteration list (multiply by iterations per epoch)
  local iteration_list=()
  local epoch_list=()
  for power in "${powers_list[@]}"; do
    local iter=$((power * ITERATIONS_PER_EPOCH))
    iteration_list+=($iter)
    epoch_list+=($power)
  done
  
  # Add the very last checkpoint if it's not already included
  if [[ ! " ${iteration_list[@]} " =~ " ${last_iteration} " ]]; then
    iteration_list+=($last_iteration)
    epoch_list+=($((last_iteration / ITERATIONS_PER_EPOCH)))
  fi

  # Make sure we have at least one iteration
  if [ ${#iteration_list[@]} -eq 0 ]; then
    log_message "Warning: No iterations to process. Exiting."
    return 0
  fi

  # Display iterations and epochs clearly
  log_message "Iterations to be converted: $(print_array_formatted iteration_list '%d')"
  log_message "Corresponding to epochs: $(print_array_formatted epoch_list '%d')"

  # Count already converted iterations (only if not in dry run mode)
  local already_converted=0
  if ! $DRY_RUN; then
    for iteration in "${iteration_list[@]}"; do
      if is_already_converted "$iteration" "$expr_dir"; then
        already_converted=$((already_converted + 1))
      fi
    done
    
    if [ "$already_converted" -gt 0 ]; then
      log_message "Found $already_converted checkpoints already converted"
    fi
  fi

  # Process each iteration
  local success_count=0
  local fail_count=0
  for iteration in "${iteration_list[@]}"; do
    process_iteration "$iteration" "$iteration_file" "$checkpoint_dir" "$expr_dir"
    if [ $? -eq 0 ]; then
      success_count=$((success_count + 1))
    else
      fail_count=$((fail_count + 1))
    fi
  done

  # Restore the original value
  if $DRY_RUN; then
    log_message "[DRY RUN] Would restore original checkpoint iteration: $last_iteration (epoch $last_epoch)"
  else
    log_message "Restoring original checkpoint iteration: $last_iteration (epoch $last_epoch)"
    update_iteration_file "$iteration_file" "$last_iteration"
  fi

  if $DRY_RUN; then
    log_message "[DRY RUN] All checkpoints would be processed."
    log_message "[DRY RUN] Summary: $success_count would succeed, $fail_count would fail"
  else
    log_message "All checkpoints processed."
    log_message "Summary: $success_count successful, $fail_count failed, $already_converted skipped"
  fi
  
  return 0
}

# Execute main function
main "$@"
exit $?