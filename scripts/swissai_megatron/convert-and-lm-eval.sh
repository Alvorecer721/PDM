#!/bin/bash

# Clones lm-eval if not exists to scratch, updates the repo and installs it.
# For local Megatron checkpoints: converts to torch dist and HF checkpoints if not already done.
# For HuggingFace model identifiers: skips conversion and uses model directly from HF Hub.
# Finally runs lm-eval on the model.
# ATTENTION:
# this script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Default values
DEFAULT_MODEL_TYPE="llama3"
DEFAULT_TOKENIZER="meta-llama/Llama-3.2-3B"
DEFAULT_TASKS="hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"
DEFAULT_BATCH_SIZE="16"
DEFAULT_MAX_LENGTH=""  # Empty means no max_length constraint
DEFAULT_APPLY_CHAT_TEMPLATE="false"
DEFAULT_OFFLINE_DATASETS="true"
DEFAULT_NO_CONVERT="false"
DEFAULT_WANDB_ENABLED="true"
DEFAULT_WANDB_PROJECT="lm-eval"

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
    echo "  --model-type TYPE             Model architecture type (default: ${DEFAULT_MODEL_TYPE})"
    echo "                                Options: llama3, apertus"
    echo "  --tokenizer TOKENIZER         Tokenizer to use (default: model-type specific)"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --batch-size SIZE             Batch size for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit)"
    echo "  --apply-chat-template         Apply chat template to inputs (default: ${DEFAULT_APPLY_CHAT_TEMPLATE})"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    echo "  --no-convert                  Skip model conversion (expects HF directory to exist)"
    echo "  --no-wandb                    Disable wandb logging (default: enabled)"
    echo "  --wandb-project PROJECT       Wandb project name (default: ${DEFAULT_WANDB_PROJECT})"
    echo "  --group-name NAME             Group name for wandb run naming and tags"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL_TYPE="${DEFAULT_MODEL_TYPE}"
TOKENIZER="${DEFAULT_TOKENIZER}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"
NO_CONVERT="${DEFAULT_NO_CONVERT}"
WANDB_ENABLED="${DEFAULT_WANDB_ENABLED}"
WANDB_PROJECT="${DEFAULT_WANDB_PROJECT}"
GROUP_NAME=""
TOKENIZER_EXPLICITLY_SET="false"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            TOKENIZER_EXPLICITLY_SET="true"
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
        --no-convert)
            NO_CONVERT="true"
            shift
            ;;
        --no-wandb)
            WANDB_ENABLED="false"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --group-name)
            GROUP_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set model-type specific defaults if tokenizer not explicitly set
if [ "$TOKENIZER_EXPLICITLY_SET" = "false" ]; then
    if [ "$MODEL_TYPE" = "apertus" ]; then
        TOKENIZER="alehc/swissai-tokenizer"
    else
        TOKENIZER="meta-llama/Llama-3.2-3B"
    fi
fi

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
    if [ "$NO_CONVERT" = "true" ]; then
        # When --no-convert is used, pass the path as-is (don't append /HF)
        MODEL_PATH="${EXPR_PATH}"
        echo "Using experiment path as-is (no /HF suffix): ${EXPR_PATH}"
    else
        # When conversion is performed, append /HF to the path
        MODEL_PATH="${EXPR_PATH}/HF"
        echo "Detected local Megatron experiment path: ${EXPR_PATH}"
    fi
fi

# Print configuration
echo "========================================"
echo "LM Evaluation Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "Is HF identifier:     ${IS_HF_IDENTIFIER}"
echo "Model path:           ${MODEL_PATH}"
echo "Model type:           ${MODEL_TYPE}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Apply chat template:  ${APPLY_CHAT_TEMPLATE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "No convert:           ${NO_CONVERT}"
echo "Wandb enabled:        ${WANDB_ENABLED}"
echo "Wandb project:        ${WANDB_PROJECT}"
echo "Group name:           ${GROUP_NAME}"
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
    # Strip trailing /HF or /HF/ so result dir uses the actual experiment name
    CLEAN_PATH="${EXPR_PATH%/}"    # remove trailing slash
    CLEAN_PATH="${CLEAN_PATH%/HF}" # remove trailing /HF
    EXPR_NAME=$(basename ${CLEAN_PATH})
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
fi

# Pin to a stable version that works with transformers 4.48.2
echo "Checking out stable lm-evaluation-harness version..."
cd "$EVAL_DIR" || exit
git fetch
# Use v0.4.9.1 tag which is compatible with transformers 4.48.2
# This version doesn't pass dtype to model initialization
git checkout v0.4.9.1 2>/dev/null || git checkout tags/v0.4.9.1 2>/dev/null || echo "⚠️  Could not checkout v0.4.9, using current version"

# Create results directory
mkdir -p "${RES_PATH}"

# Run conversion only for local Megatron checkpoints
if [ "$IS_HF_IDENTIFIER" = "true" ]; then
    echo "Skipping conversion for HuggingFace model identifier..."
elif [ "$NO_CONVERT" = "true" ]; then
    echo "Skipping conversion as requested (--no-convert)..."
    # Validate HF directory exists
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "ERROR: HF directory does not exist: ${MODEL_PATH}"
        echo "       Please run conversion first or remove --no-convert flag"
        exit 1
    fi
    echo "HF directory found: ${MODEL_PATH}"
else
    # Run conversion using shared script
    echo "Running checkpoint conversion..."

    # Build conversion arguments
    CONVERT_ARGS="${EXPR_PATH} --model-type ${MODEL_TYPE}"
    if [ "$TOKENIZER_EXPLICITLY_SET" = "true" ]; then
        CONVERT_ARGS="${CONVERT_ARGS} --tokenizer ${TOKENIZER}"
    fi

    bash ${PDM_DIR}/scripts/swissai_megatron/convert.sh ${CONVERT_ARGS}

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

# Pin PEFT to compatible version for transformers 4.48.2 in NGC container
echo "📦 Pinning PEFT to compatible version..."
pip install "peft==0.13.2"
pip install --upgrade "transformers>=4.56,<5.0.0"

# Install wandb extra if wandb logging is enabled
if [ "$WANDB_ENABLED" = "true" ]; then
    echo "📦 Installing wandb support for lm-eval..."
    pip install "lm_eval[wandb]"
fi

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
MODEL_ARGS="pretrained=${MODEL_PATH}"
# Only add tokenizer if explicitly set OR if it's not a HF identifier
if [ "$TOKENIZER_EXPLICITLY_SET" = "true" ] || [ "$IS_HF_IDENTIFIER" = "false" ]; then
    MODEL_ARGS="${MODEL_ARGS},tokenizer=${TOKENIZER}"
fi
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LENGTH}"
fi
APPLY_CHAT=""
if [ "$APPLY_CHAT_TEMPLATE" = "true" ]; then
    APPLY_CHAT="--apply_chat_template --fewshot_as_multiturn"
fi

# Run evaluation command
echo "Running LM evaluation..."
# Conditionally set/unset offline mode
if [ "$OFFLINE_DATASETS" = "true" ]; then
    export HF_DATASETS_OFFLINE=1
else
    unset HF_DATASETS_OFFLINE
fi

# Build wandb arguments if enabled
WANDB_FLAG=""
if [ "$WANDB_ENABLED" = "true" ]; then
    # Build run name: {EXPR_NAME}_{GROUP_NAME} or just {EXPR_NAME}
    if [ -n "$GROUP_NAME" ]; then
        WANDB_RUN_NAME="${EXPR_NAME}_${GROUP_NAME}"
    else
        WANDB_RUN_NAME="${EXPR_NAME}"
    fi

    # Build tags via WANDB_TAGS env var (wandb reads this natively as comma-separated list)
    # Cannot pass tags in --wandb_args because commas in tags conflict with the key=value,key=value format
    # Note: wandb tags must be <= 64 characters each, so we use short tags only (group name + task names)
    export WANDB_TAGS=""
    if [ -n "$GROUP_NAME" ]; then
        export WANDB_TAGS="${GROUP_NAME}"
    fi
    # Add individual task names as tags
    IFS=',' read -ra TAG_TASKS <<< "$TASKS"
    for t in "${TAG_TASKS[@]}"; do
        if [ -n "$WANDB_TAGS" ]; then
            export WANDB_TAGS="${WANDB_TAGS},${t}"
        else
            export WANDB_TAGS="${t}"
        fi
    done

    WANDB_FLAG="--wandb_args project=${WANDB_PROJECT},name=${WANDB_RUN_NAME}"
    echo "Wandb run name: ${WANDB_RUN_NAME}"
    echo "Wandb tags (env): ${WANDB_TAGS}"
fi

accelerate launch ${PDM_DIR}/scripts/swissai_megatron/wandb_guard_launcher.py -m lm_eval --model hf \
   --model_args "${MODEL_ARGS}" \
   --tasks "${TASKS}" \
   --batch_size "${BATCH_SIZE}" \
   --output_path "${RES_PATH}" \
   ${APPLY_CHAT} \
   ${WANDB_FLAG}

echo "Evaluation completed. Results saved to: ${RES_PATH}"
