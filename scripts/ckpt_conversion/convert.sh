#!/bin/bash

# Function to handle errors
# $1 - Error message to display
# >&2 redirects output to stderr (standard error)
error_exit() {
    echo "ERROR: $1" >&2   # Print error message to stderr
    exit 1                  # Exit script with error status 1
}

# Function to print success messages
# $1 - Success message to display
print_success() {
    echo "SUCCESS: $1"
}

# Validate command line arguments
# $# - Number of arguments passed to script
# -ne - "not equal to"
# This function ensures exactly one argument is provided
validate_args() {
    if [ "$#" -ne 1 ]; then    # If number of arguments is not equal to 1
        error_exit "Usage: $0 /path/to/checkpoint/megatron_ckpt_folder"
        # $0 represents the script name itself
    fi
}

# Setup and validate all required paths and directories
# $1 - Path to checkpoint file passed as argument
setup_paths() {
    local checkpoint_path="$1"    # 'local' declares variable with local scope
    
    # -f tests if file exists and is a regular file
    # [ -f "$checkpoint_path" ] || error_exit "Checkpoint file does not exist: $checkpoint_path"
    
    # dirname extracts the directory path from a file path
    # basename extracts the filename from a file path
    CHECKPOINT_DIR=$(dirname "$checkpoint_path")
    FILENAME=$(basename "$checkpoint_path")
    
    # Get experiment directory by going up two levels
    # $(command) executes command and returns its output
    EXPERIMENT_DIR="$(dirname "$(dirname "${CHECKPOINT_DIR}")")"

    echo "Base directory path: $EXPERIMENT_DIR" 
    
    # Setup output directories and config files
    # These are global variables (uppercase by convention)
    NEMO_OUTPUT_DIR="$EXPERIMENT_DIR/results/Megatron2NeMo"
    HF_OUTPUT_DIR="$EXPERIMENT_DIR/results/NeMo2HF"
    HPARAMS_FILE="$EXPERIMENT_DIR/results/hparams.yaml"
    OVERRIDE_CONFIG_PATH="$EXPERIMENT_DIR/results/nemo-toHF-config.yaml"
    
    # Check if hparams file exists
    [ -f "$HPARAMS_FILE" ] || error_exit "hparams.yaml file not found at: $HPARAMS_FILE"
    
    # Create output directories
    # -p creates parent directories if they don't exist
    mkdir -p "$NEMO_OUTPUT_DIR" "$HF_OUTPUT_DIR"
    
    # Create override config by replacing 'cfg:' with 'model:'
    # 's/pattern/replacement/' is sed's substitute command
    sed 's/^cfg:/model:/' "$HPARAMS_FILE" > "$OVERRIDE_CONFIG_PATH"
}

# Extract step and consumed_samples information from checkpoint filename
# $1 - Checkpoint filename
extract_checkpoint_info() {
    local filename="$1"
    
    # Use sed to extract numbers after 'step=' and before '-'
    # -n suppresses automatic printing
    # p prints the matched pattern
    STEPS=$(echo "$filename" | sed -n 's/.*step=\([0-9]*\)-.*/\1/p')
    
    # Extract numbers after 'consumed_samples=' and before '.0'
    CONSUMED=$(echo "$filename" | sed -n 's/.*consumed_samples=\([0-9]*\).0/\1/p')
    
    # -z tests if string is empty
    if [ -z "$STEPS" ] || [ -z "$CONSUMED" ]; then
        error_exit "Could not extract step and consumed_samples from filename: $filename"
    fi
    
    # Construct output paths using extracted information
    NEMO_FILE_PATH="$NEMO_OUTPUT_DIR/step=$STEPS-consumed=$CONSUMED.nemo"
    HF_FILE_PATH="$HF_OUTPUT_DIR/step=$STEPS-consumed=$CONSUMED.bin"
}

# Convert Megatron checkpoint to NeMo format
convert_to_nemo() {
    # Check if output file already exists
    if [ -f "$NEMO_FILE_PATH" ]; then
        echo "Nemo checkpoint already exists: $NEMO_FILE_PATH"
        return 0    # Exit function with success status
    fi

    echo "Converting Megatron ckpt to Nemo ckpt: $FILENAME"
    
    # Launch distributed Python process for conversion
    # -m torch.distributed.launch: Launch script in distributed mode
    # --nproc_per_node=4: Use 4 processes (GPUs)
    python3 -m torch.distributed.launch --nproc_per_node=4 \
        ${NEMO_REPO_DIR}/examples/nlp/language_modeling/megatron_ckpt_to_nemo.py \
        --checkpoint_folder "$CHECKPOINT_DIR" \
        --checkpoint_name "$FILENAME" \
        --nemo_file_path "$NEMO_FILE_PATH" \
        --model_type gpt \
        --hparams_file "$HPARAMS_FILE" \
        --tensor_model_parallel_size 1 \
        --pipeline_model_parallel_size 1 \
        --gpus_per_node 4

    # $? contains exit status of last command
    # -eq 0 checks if exit status equals 0 (success)
    if [ $? -eq 0 ]; then
        print_success "Megatron to NeMo conversion completed successfully"
        echo "Output saved to: $NEMO_FILE_PATH"
    else
        error_exit "Megatron to NeMo conversion failed"
    fi
}

# Convert NeMo checkpoint to Hugging Face format
convert_to_hf() {
    # Check if output file already exists
    if [ -f "$HF_FILE_PATH" ]; then
        echo "Hugging Face checkpoint already exists: $HF_FILE_PATH"
        return 0    # Exit function with success status
    fi

    echo "Converting NeMo ckpt to Hugging Face ckpt: $NEMO_FILE_PATH"
    
    # Run Python script for NeMo to HF conversion
    # -u: Force stdout to be unbuffered
    python3 -u ${NEMO_REPO_DIR}/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py \
        --input_name_or_path="$NEMO_FILE_PATH" \
        --output_path="$HF_FILE_PATH" \
        --override_config_path="$OVERRIDE_CONFIG_PATH"

    if [ $? -eq 0 ]; then
        print_success "NeMo to Hugging Face conversion completed successfully"
        echo "Output saved to: $HF_FILE_PATH"
    else
        error_exit "NeMo to Hugging Face conversion failed"
    fi
}

# Print summary of all conversions
print_summary() {
    print_success "All conversions completed for checkpoint: $FILENAME"
    echo "NeMo checkpoint: $NEMO_FILE_PATH"
    echo "Hugging Face checkpoint: $HF_FILE_PATH"
}

# Main execution function
# "$@" passes all command line arguments
main() {
    validate_args "$@"           # Validate input arguments
    setup_paths "$1"            # Setup necessary paths ($1 is first argument)
    extract_checkpoint_info "$FILENAME"  # Extract checkpoint information
    convert_to_nemo            # Convert to NeMo format
    convert_to_hf              # Convert to Hugging Face format
    print_summary             # Print final summary
}

# Start script execution
# "$@" passes all command line arguments to main
main "$@"
