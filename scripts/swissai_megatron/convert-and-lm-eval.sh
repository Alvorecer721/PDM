#!/bin/bash

# Clones lm-eval if not exists to scratch, updates the repo and installs it.
# Converts to torch dist and HF checkpoints if not already done.
# Finally runs lm-eval on the converted HF model.
# ATTENTION:
# this script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Default values
DEFAULT_TOKENIZER="meta-llama/Llama-3.2-3B"
DEFAULT_TASKS="hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"
DEFAULT_BATCH_SIZE=4
DEFAULT_MAX_LENGTH=""  # Empty means no max_length constraint
DEFAULT_APPLY_CHAT_TEMPLATE="false"
DEFAULT_OFFLINE_DATASETS="true"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory [OPTIONS]"
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

# Print configuration
echo "========================================"
echo "LM Evaluation Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Tasks:                ${TASKS}"
echo "Batch size(per GPU):  ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Apply chat template:  ${APPLY_CHAT_TEMPLATE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "Num GPU:              ${NUM_GPUS}"
echo "Global Batch Size     $((NUM_GPUS * BATCH_SIZE))"
echo "========================================"
echo ""

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lm-evaluation-harness
PDM_DIR=/iopsstor/scratch/cscs/$USER/PDM

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

EXPR_NAME=$(basename ${EXPR_PATH})
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

# Run conversion using shared script
echo "Running checkpoint conversion..."
bash ${PDM_DIR}/scripts/swissai_megatron/convert.sh "${EXPR_PATH}"

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion failed"
   exit 1
fi

# Install/update lm-eval
echo "📦 Setting up lm-eval package..."
cd "$EVAL_DIR" || exit
pip install -e .

# Build model_args
MODEL_ARGS="pretrained=${EXPR_PATH}/HF,tokenizer=${TOKENIZER}"
if [ -n "$MAX_LENGTH" ]; then
    MODEL_ARGS="${MODEL_ARGS},max_length=${MAX_LENGTH}"
fi
if [ "$APPLY_CHAT_TEMPLATE" = "true" ]; then
    MODEL_ARGS="${MODEL_ARGS},apply_chat_template=true"
fi

# Run evaluation command
echo "Running LM evaluation..."
# Conditionally set offline mode
[ "$OFFLINE_DATASETS" = "true" ] && export HF_DATASETS_OFFLINE=1

accelerate launch --num_processes="${NUM_GPUS}" -m lm_eval --model hf \
   --model_args "${MODEL_ARGS}" \
   --tasks "${TASKS}" \
   --batch_size "${BATCH_SIZE}" \
   --output_path "${RES_PATH}"

echo "Evaluation completed. Results saved to: ${RES_PATH}"
