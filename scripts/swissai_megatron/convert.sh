#!/bin/bash

# Unified conversion script that dispatches to model-specific converters
# Supports both Llama3 and Apertus model types
#
# Path arguments:
#   experiment_path  - Output directory for converted models (torch/, HF/)
#   checkpoint_path  - Source torch-dist checkpoint (default: ${experiment_path}/checkpoints/3B)
#
# Usage: convert.sh <experiment_path> [--model-type TYPE] [--checkpoint-path PATH] [--tokenizer TOKENIZER]
#
# ATTENTION:
# This script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Default values
DEFAULT_MODEL_TYPE="llama3"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory [--model-type TYPE] [--checkpoint-path PATH] [--tokenizer TOKENIZER]"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE        Model architecture type (default: ${DEFAULT_MODEL_TYPE})"
    echo "                           Options: llama3, apertus"
    echo "  --checkpoint-path PATH   Direct path to torch distributed checkpoint"
    echo "                           (default: \${experiment_path}/checkpoints/3B)"
    echo "  --tokenizer TOKENIZER    HuggingFace tokenizer path or identifier"
    echo "                           (optional, model-type specific defaults apply)"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL_TYPE="${DEFAULT_MODEL_TYPE}"
TOKENIZER=""
CHECKPOINT_PATH=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default checkpoint path if not specified
if [ -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="${EXPR_PATH}/checkpoints/3B"
fi

# Validate model type
if [ "$MODEL_TYPE" != "llama3" ] && [ "$MODEL_TYPE" != "apertus" ]; then
    echo "ERROR: Invalid model type: ${MODEL_TYPE}"
    echo "Valid options: llama3, apertus"
    exit 1
fi

PDM_DIR=/iopsstor/scratch/cscs/$USER/PDM
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

EXPR_NAME=$(basename ${EXPR_PATH})

echo "========================================"
echo "Unified Conversion Script"
echo "========================================"
echo "Experiment path:  ${EXPR_PATH}"
echo "Experiment name:  ${EXPR_NAME}"
echo "Checkpoint path:  ${CHECKPOINT_PATH}"
echo "Model type:       ${MODEL_TYPE}"
if [ -n "$TOKENIZER" ]; then
    echo "Tokenizer:        ${TOKENIZER}"
else
    echo "Tokenizer:        (model-type default)"
fi
echo "========================================"
echo ""

# Dispatch to appropriate converter
if [ "$MODEL_TYPE" = "llama3" ]; then
    echo "Using Llama3 conversion pipeline..."

    # Check if torch directory already exists, skip conversion if it does
    if [ -d "${EXPR_PATH}/torch" ]; then
        echo "Torch directory already exists, skipping distributed to torch conversion..."
    else
        # Run the torch distributed to torch conversion
        echo "Converting torch distributed to torch..."
        CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun ${PDM_DIR}/src/convert/convert_torch_dist_to_torch.py \
            --bf16 \
            --load "${CHECKPOINT_PATH}" \
            --ckpt-convert-save "${EXPR_PATH}"

        # Check if the conversion was successful
        if [ $? -ne 0 ]; then
            echo "Torch distributed to torch conversion failed"
            exit 1
        fi
    fi

    # Here we can assume that needed troch/ cp is under experiment path
    python ${PDM_DIR}/src/convert/convert_megatron_to_hf.py \
       --experiment-path "${EXPR_PATH}"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
       echo "Model conversion failed"
       exit 1
    fi

elif [ "$MODEL_TYPE" = "apertus" ]; then
    echo "Using Apertus conversion pipeline..."

    APERTUS_ARGS="${CHECKPOINT_PATH} --output-dir ${EXPR_PATH}"
    if [ -n "$TOKENIZER" ]; then
        APERTUS_ARGS="${APERTUS_ARGS} --tokenizer ${TOKENIZER}"
    fi

    # Run conversion, store to EXPR_PATH/HF
    bash ${PDM_DIR}/scripts/swissai_megatron/convert_apertus.sh ${APERTUS_ARGS}

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
       echo "Apertus model conversion failed"
       exit 1
    fi
fi

echo ""
echo "Conversion completed successfully!"
