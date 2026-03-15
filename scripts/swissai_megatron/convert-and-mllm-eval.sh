#!/bin/bash
set -e  # Exit on error

# Clones lmms-eval if not exists to scratch, updates the repo and installs it.
# For local Megatron checkpoints: converts to torch dist and HF checkpoints if not already done.
# For HuggingFace model identifiers: skips conversion and uses model directly from HF Hub.
# Finally runs lmms-eval on the model.
# ATTENTION:
# this script should be run inside computing node with:
# --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml

# Unset SSL_CERT_FILE to avoid FileNotFoundError inside NGC container
# (host path /etc/ssl/ca-bundle.pem doesn't exist in container; Python falls back to defaults)
unset SSL_CERT_FILE

# Default values (current as-is status)
DEFAULT_MODEL_TYPE="llama3"
DEFAULT_MODEL="llama_emu3p5"
DEFAULT_TASKS="ai2d"
DEFAULT_BATCH_SIZE="1"
DEFAULT_MAX_LENGTH=""  # Empty means no max_length constraint
DEFAULT_OFFLINE_DATASETS="true"
DEFAULT_NO_CONVERT="false"
DEFAULT_EMU_MIN_PIXELS="16384"   # 128*128
DEFAULT_EMU_MAX_PIXELS="1048576" # 1024*1024
DEFAULT_WANDB_ENABLED="true"
DEFAULT_WANDB_PROJECT="lmms-eval"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE             Model architecture type (default: ${DEFAULT_MODEL_TYPE})"
    echo "                                Options: llama3, apertus"
    echo "  --tokenizer TOKENIZER         Tokenizer to use (optional, model-type specific default)"
    echo "  --model MODEL                 MLLM model type (default: ${DEFAULT_MODEL})"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --batch-size SIZE             Batch size per GPU for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit, will be set internally)"
    echo "  --emu-min-pixels PIXELS      Minimum pixels for EMU vision encoder (e.g., 262144 for 512*512)"
    echo "  --emu-max-pixels PIXELS      Maximum pixels for EMU vision encoder (e.g., 1048576 for 1024*1024)"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    echo "  --no-convert                  Skip model conversion (expects HF directory to exist)"
    echo "  --old-megatron                Use legacy pretrain_gpt model_provider for conversion (deprecated)"
    echo "  --no-wandb                    Disable wandb logging (default: enabled)"
    echo "  --wandb-project PROJECT       Wandb project name (default: ${DEFAULT_WANDB_PROJECT})"
    echo "  --group-name NAME             Group name for wandb tags"
    echo "  --debug                       Adds debug paramters to lmms eval run so first 5 samples are run with debug prints"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL_TYPE="${DEFAULT_MODEL_TYPE}"
TOKENIZER=""
MODEL="${DEFAULT_MODEL}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"
NO_CONVERT="${DEFAULT_NO_CONVERT}"
EMU_MIN_PIXELS="${DEFAULT_EMU_MIN_PIXELS}"
EMU_MAX_PIXELS="${DEFAULT_EMU_MAX_PIXELS}"
TOKENIZER_EXPLICITLY_SET="false"
OLD_MEGATRON="false"
DEBUG=""
WANDB_ENABLED="${DEFAULT_WANDB_ENABLED}"
WANDB_PROJECT="${DEFAULT_WANDB_PROJECT}"
GROUP_NAME=""

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
        --debug)
            DEBUG="true"
            shift
            ;;
        --model)
            MODEL="$2"
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
        --emu-min-pixels)
            EMU_MIN_PIXELS="$2"
            shift 2
            ;;
        --emu-max-pixels)
            EMU_MAX_PIXELS="$2"
            shift 2
            ;;
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        --no-convert)
            NO_CONVERT="true"
            shift
            ;;
        --old-megatron)
            OLD_MEGATRON="true"
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
echo "MLLM Evaluation Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "Is HF identifier:     ${IS_HF_IDENTIFIER}"
echo "Model path:           ${MODEL_PATH}"
echo "Model type:           ${MODEL_TYPE}"
echo "MLLM model:           ${MODEL}"
if [ -n "$TOKENIZER" ]; then
    echo "Tokenizer:            ${TOKENIZER}"
fi
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "No convert:           ${NO_CONVERT}"
echo "Wandb enabled:        ${WANDB_ENABLED}"
echo "Wandb project:        ${WANDB_PROJECT}"
echo "Group name:           ${GROUP_NAME}"
echo "Num GPU:              ${NUM_GPUS}"
echo "========================================"
echo ""

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval
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
    EXPR_NAME=$(basename "${CLEAN_PATH}")
fi
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/${EXPR_NAME}"

if [ ! -d "$EVAL_DIR" ]; then
    echo "Creating lmms-eval directory at $EVAL_DIR as it doesnt exist yet..."
    git clone https://github.com/swiss-ai/lmms-eval.git "$EVAL_DIR"
else
    echo "lmms-eval directory exists, updating repository..."
    cd "$EVAL_DIR" || exit
    git pull
    git status
fi

# Create results directory
mkdir -p "${RES_PATH}"

# Inform user about results location
echo "========================================"
echo "Results Configuration"
echo "========================================"
echo "Results will be stored in:"
echo "  ${RES_PATH}"
echo ""
echo "NOTE: Existing results will NOT be overwritten."
echo "      If results already exist, lmms-eval will skip"
echo "      completed evaluations or may append to results."
echo "========================================"
echo ""

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
elif [ -d "${MODEL_PATH}" ]; then
    echo "HF directory already exists, skipping conversion: ${MODEL_PATH}"
else
    echo "Running checkpoint conversion..."

    # Build conversion arguments
    CONVERT_ARGS=("${EXPR_PATH}" --model-type "${MODEL_TYPE}")
    if [ "$TOKENIZER_EXPLICITLY_SET" = "true" ]; then
        CONVERT_ARGS+=(--tokenizer "${TOKENIZER}")
    fi
    if [ "$OLD_MEGATRON" = "true" ]; then
        CONVERT_ARGS+=(--old-megatron)
    fi

    bash "${PDM_DIR}/scripts/swissai_megatron/convert.sh" "${CONVERT_ARGS[@]}"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
       echo "Model conversion failed"
       exit 1
    fi
fi

# Install/update lmms-eval
echo "📦 Setting up lmms-eval package..."
cd "$EVAL_DIR" || exit
pip uninstall jupyterlab -y 2>/dev/null || true  # Uninstall conflicting package (may not be installed)

# Install lmms-eval with all dependencies from pyproject.toml
pip install -e .

# Install optional dependencies that are commonly needed for evaluations
echo "📦 Installing optional dependencies from pyproject.toml..."
pip install -e ".[metrics]"

# Reinstall core dependencies from pyproject.toml to fix version conflicts with NGC container
# The initial installation may have downgraded some packages due to dependency resolution
# This ensures versions are compatible with both NGC container and lmms-eval requirements
echo "📦 Fixing dependency conflicts by reinstalling from pyproject.toml..."

# Emu3/Emu3.5 baseline models use a custom modeling_emu3.py that relies on transformers <4.50 APIs:
#   - PreTrainedModel inheriting GenerationMixin (removed in v4.50)
#   - DynamicCache.seen_tokens (renamed in v4.50)
# Pin transformers <4.50 for these models to avoid cascading compatibility issues.
if [[ "$MODEL" == "emu3p5" ]]; then
    TRANSFORMERS_SPEC="transformers==4.48.2"
    echo "  Pinning transformers==4.48.2 for emu3p5 baseline model compatibility"
elif [[ "$MODEL" == "emu3" ]]; then
    TRANSFORMERS_SPEC="transformers==4.44.0"
    echo "  Pinning transformers==4.44.0 for emu3 baseline model compatibility"
else
    TRANSFORMERS_SPEC="transformers>=4.56,<5.0.0"
fi

pip install --upgrade \
    "accelerate>=0.29.1" \
    "datasets>=3.0.0" \
    "${TRANSFORMERS_SPEC}" \
    "peft>=0.2.0" \
    "numpy>=1.26.4" \
    "pillow" \
    "tiktoken"
    #"torch>=2.1.0" \
    #"torchvision>=0.16.0" \

# Fix specific version conflicts with NGC container packages
# The lmms-eval installation may have downgraded antlr4-python3-runtime which breaks hydra-core and omegaconf
echo "📦 Fixing antlr4-python3-runtime and omegaconf version conflict..."
pip uninstall antlr4-python3-runtime omegaconf -y 2>/dev/null || true
pip install --force-reinstall --no-cache-dir "antlr4-python3-runtime>=4.9,<5.0" "omegaconf>=2.3.0"

# Pin PEFT to compatible version for transformers in NGC container
#echo "📦 Pinning PEFT to compatible version..."
#pip install "peft==0.13.2"

# Install wandb if wandb logging is enabled
if [ "$WANDB_ENABLED" = "true" ]; then
    echo "📦 Installing wandb for lmms-eval logging..."
    pip install wandb
fi

# Install task-specific dependencies
echo "📦 Installing task-specific dependencies..."
if [ -f "${EVAL_DIR}/examples/install_task_deps.sh" ]; then
    source "${EVAL_DIR}/examples/install_task_deps.sh" "${TASKS}" "${EVAL_DIR}"
else
    echo "Warning: Task dependency installation script not found at ${EVAL_DIR}/examples/install_task_deps.sh"
fi

# Build model_args
# Use model_descriptor and tokenizer_path (required by llama_emu3 and similar models)
# If tokenizer is not explicitly set, use the model path as tokenizer path
if [ -n "$TOKENIZER" ]; then
    TOKENIZER_PATH="$TOKENIZER"
else
    TOKENIZER_PATH="${MODEL_PATH}"
fi

MODEL_ARGS="model_descriptor=${MODEL_PATH},tokenizer_path=${TOKENIZER_PATH}"
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LENGTH}"
fi
if [ -n "$EMU_MIN_PIXELS" ]; then
    MODEL_ARGS="${MODEL_ARGS},emu_min_pixels=${EMU_MIN_PIXELS}"
fi
if [ -n "$EMU_MAX_PIXELS" ]; then
    MODEL_ARGS="${MODEL_ARGS},emu_max_pixels=${EMU_MAX_PIXELS}"
fi

# Run evaluation command
echo "Running MLLM evaluation..."
# Conditionally set/unset offline mode
if [ "$OFFLINE_DATASETS" = "true" ]; then
    export HF_DATASETS_OFFLINE=1
else
    unset HF_DATASETS_OFFLINE
fi

# Build wandb arguments if enabled
WANDB_FLAG=""
if [ "$WANDB_ENABLED" = "true" ]; then
    # Use experiment name as run name (group manually in wandb UI)
    WANDB_RUN_NAME="${EXPR_NAME}"

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
    # Add emu pixel range tags if set
    if [ -n "$EMU_MIN_PIXELS" ]; then
        export WANDB_TAGS="${WANDB_TAGS},emu_min_px=${EMU_MIN_PIXELS}"
    fi
    if [ -n "$EMU_MAX_PIXELS" ]; then
        export WANDB_TAGS="${WANDB_TAGS},emu_max_px=${EMU_MAX_PIXELS}"
    fi

    WANDB_FLAG="--wandb_args project=${WANDB_PROJECT},name=${WANDB_RUN_NAME}"
    echo "Wandb run name: ${WANDB_RUN_NAME}"
    echo "Wandb tags (env): ${WANDB_TAGS}"
fi

DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
    DEBUG_FLAG="--debug"
fi

accelerate launch ${PDM_DIR}/scripts/swissai_megatron/wandb_guard_launcher.py -m lmms_eval \
    --model "${MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${RES_PATH}" \
    ${DEBUG_FLAG} \
    ${WANDB_FLAG}

echo "Evaluation completed. Results saved to: ${RES_PATH}"
