#!/bin/bash

# Clones lm-eval if not exists to scratch, updates the repo and installs it.
# For local Megatron checkpoints: converts to torch dist and HF checkpoints if not already done.
# For HuggingFace model identifiers: skips conversion and uses model directly from HF Hub.
# Finally runs lm-eval on the model.
# ATTENTION:
# this script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Default values
DEFAULT_TOKENIZER="meta-llama/Llama-3.2-3B"
DEFAULT_TASKS="hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"
DEFAULT_BATCH_SIZE="16"
DEFAULT_MAX_LENGTH=""  # Empty means no max_length constraint
DEFAULT_APPLY_CHAT_TEMPLATE="false"
DEFAULT_OFFLINE_DATASETS="true"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide experiment path or HuggingFace model identifier"
    echo "Usage: $0 <experiment_path|hf_model_id> [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/megatron/experiment"
    echo "  $0 meta-llama/Llama-3.2-3B --instruct"
    echo ""
    echo "Options:"
    echo "  --tokenizer TOKENIZER         Tokenizer to use (default: ${DEFAULT_TOKENIZER})"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --batch-size SIZE             Batch size for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit)"
    echo "  --apply-chat-template         Apply chat template to inputs (default: ${DEFAULT_APPLY_CHAT_TEMPLATE})"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
TOKENIZER="${DEFAULT_TOKENIZER}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --apply-chat-template)
            APPLY_CHAT_TEMPLATE="true"
            shift
            ;;
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Automatically get number of available GPUS to configure accelerate properöy
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Number of GPUs available: $NUM_GPUS"
echo "GPUS: $CUDA_VISIBLE_DEVICES"

# Detect if EXPR_PATH is a HuggingFace identifier or a local path
# HF identifiers typically have format: org/model or model
# Local paths start with / or . or are existing directories
IS_HF_IDENTIFIER="false"
if [[ ! "$EXPR_PATH" =~ ^[/.] ]] && [[ ! -d "$EXPR_PATH" ]]; then
    # Looks like a HF identifier (doesn't start with / or . and is not an existing directory)
    IS_HF_IDENTIFIER="true"
    MODEL_PATH="$EXPR_PATH"
    echo "Detected HuggingFace model identifier: ${EXPR_PATH}"
else
    # Local Megatron experiment path
    MODEL_PATH="${EXPR_PATH}/HF"
    echo "Detected local Megatron experiment path: ${EXPR_PATH}"
fi

# Print configuration
echo "========================================"
echo "LM Evaluation Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "Is HF identifier:     ${IS_HF_IDENTIFIER}"
echo "Model path:           ${MODEL_PATH}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Apply chat template:  ${APPLY_CHAT_TEMPLATE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Num GPU:              ${NUM_GPUS}"
echo "========================================"
echo ""

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lm-evaluation-harness
PDM_DIR=/iopsstor/scratch/cscs/$USER/PDM

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Generate a safe experiment name for results directory
if [ "$IS_HF_IDENTIFIER" = "true" ]; then
    # Replace slashes with double underscores for HF identifiers
    EXPR_NAME=$(echo "${EXPR_PATH}" | sed 's/\//__/g')
else
    EXPR_NAME=$(basename ${EXPR_PATH})
fi
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lm_eval/${EXPR_NAME}"

# Clear existing results directory if it exists
if [ -d "${RES_PATH}" ]; then
    echo "Clearing existing results directory: ${RES_PATH}"
    rm -rf "${RES_PATH}"/*
fi

if [ ! -d "$EVAL_DIR" ]; then
    echo "Creating lm-evaluation-harness directory..."
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$EVAL_DIR"
else
    echo "lm-evaluation-harness directory exists, updating repository..."
    cd "$EVAL_DIR" || exit
    git pull
fi

# Create results directory
mkdir -p "${RES_PATH}"

# Run conversion only for local Megatron checkpoints
if [ "$IS_HF_IDENTIFIER" = "true" ]; then
    echo "Skipping conversion for HuggingFace model identifier..."
else
    # Run conversion using shared script
    echo "Running checkpoint conversion..."
    bash ${PDM_DIR}/scripts/swissai_megatron/convert.sh "${EXPR_PATH}"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
       echo "Model conversion failed"
       exit 1
    fi
fi

# Install/update lm-eval
echo "📦 Setting up lm-eval package..."
cd "$EVAL_DIR" || exit
pip install -e .

# Install optional dependencies for each task
echo "📦 Installing optional task dependencies..."
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
for task in "${TASK_ARRAY[@]}"; do
    # Remove any whitespace
    task=$(echo "$task" | xargs)
    echo "  Attempting to install dependencies for task: $task"

    # Try to install, but don't fail if extras don't exist
    if pip install -e ".[$task]" 2>&1 | grep -q "No extras are defined"; then
        echo "    ℹ️  No optional dependencies defined for '$task'"
    elif [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "    ✓ Installed dependencies for '$task'"
    else
        echo "    ⚠️  Could not install dependencies for '$task' (may not exist)"
    fi
done
echo "📦 Task dependency installation complete"

# Build model_args
MODEL_ARGS="pretrained=${MODEL_PATH},tokenizer=${TOKENIZER}"
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LENGTH}"
fi
APPLY_CHAT=""
if [ "$APPLY_CHAT_TEMPLATE" = "true" ]; then
    APPLY_CHAT="--apply_chat_template --fewshot_as_multiturn"
fi

# Run evaluation command
echo "Running LM evaluation..."
# Conditionally set offline mode
[ "$OFFLINE_DATASETS" = "true" ] && export HF_DATASETS_OFFLINE=1

accelerate launch -m lm_eval --model hf \
   --model_args "${MODEL_ARGS}" \
   --tasks "${TASKS}" \
   --batch_size "${BATCH_SIZE}" \
   --output_path "${RES_PATH}" \
   ${APPLY_CHAT}

echo "Evaluation completed. Results saved to: ${RES_PATH}"
