#!/bin/bash
set -e  # Exit on error

# Unified conversion script that dispatches to model-specific converters
# Supports both Llama3 and Apertus model types
#
# By default uses the new Megatron API (model_provider + gpt_builder).
# Pass --old-megatron to use the legacy pretrain_gpt model_provider (deprecated).
#
# Path arguments:
#   experiment_path  - Output directory for converted models (torch/, HF/)
#   checkpoint_path  - Source torch-dist checkpoint (default: ${experiment_path}/checkpoints/3B)
#
# Usage: convert.sh <experiment_path> [--model-type TYPE] [--checkpoint-path PATH] [--tokenizer TOKENIZER] [--old-megatron]
#
# ATTENTION:
# This script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Source shared utilities
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Default values
DEFAULT_MODEL_TYPE="llama3"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory [--model-type TYPE] [--checkpoint-path PATH] [--tokenizer TOKENIZER] [--old-megatron]"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE        Model architecture type (default: ${DEFAULT_MODEL_TYPE})"
    echo "                           Options: llama3, apertus"
    echo "  --checkpoint-path PATH   Direct path to torch distributed checkpoint"
    echo "                           (default: \${experiment_path}/checkpoints/3B)"
    echo "  --tokenizer TOKENIZER    HuggingFace tokenizer path or identifier"
    echo "                           (optional, model-type specific defaults apply)"
    echo "  --old-megatron           Use legacy pretrain_gpt model_provider (deprecated)"
    echo "  --logits-test            Run logits comparison after HF conversion"
    echo "  --logits-test-ref SPEC   Reference checkpoint for logits comparison"
    echo "                           (e.g. hf:/path/to/ref or meg:/path/to/ckpt)"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL_TYPE="${DEFAULT_MODEL_TYPE}"
TOKENIZER=""
CHECKPOINT_PATH=""
OLD_MEGATRON="false"
LOGITS_TEST="false"
LOGITS_TEST_REF=""

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
        --old-megatron)
            OLD_MEGATRON="true"
            shift
            ;;
        --logits-test)
            LOGITS_TEST="true"
            shift
            ;;
        --logits-test-ref)
            LOGITS_TEST_REF="$2"
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

EXPR_NAME=$(basename "${EXPR_PATH}")

echo "========================================"
echo "Unified Conversion Script"
echo "========================================"
echo "Experiment path:  ${EXPR_PATH}"
echo "Experiment name:  ${EXPR_NAME}"
echo "Checkpoint path:  ${CHECKPOINT_PATH}"
echo "Model type:       ${MODEL_TYPE}"
echo "Old Megatron:     ${OLD_MEGATRON}"
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
        detect_tp_size "${CHECKPOINT_PATH}"

        # Build conversion command as array for safe quoting
        CONVERT_ARGS=(
            "${PDM_DIR}/src/convert/convert_torch_dist_to_torch.py"
            --bf16
            --load "${CHECKPOINT_PATH}"
            --ckpt-convert-save "${EXPR_PATH}"
        )
        if [ "$OLD_MEGATRON" = "true" ]; then
            CONVERT_ARGS+=(--old-megatron)
        fi

        echo "Converting torch distributed to torch (using ${TP_SIZE} GPUs)..."
        CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node="${TP_SIZE}" "${CONVERT_ARGS[@]}"

        if [ $? -ne 0 ]; then
            echo "Torch distributed to torch conversion failed"
            exit 1
        fi
    fi

    # Here we can assume that needed torch/ cp is under experiment path
    CONVERT_HF_ARGS=(
        "${PDM_DIR}/src/convert/convert_megatron_to_hf.py"
        --experiment-path "${EXPR_PATH}"
    )
    if [ -n "$TOKENIZER" ]; then
        CONVERT_HF_ARGS+=(--tokenizer "${TOKENIZER}")
    fi

    python "${CONVERT_HF_ARGS[@]}"

    if [ $? -ne 0 ]; then
       echo "Model conversion failed"
       exit 1
    fi

    # --- Optional logits comparison for llama3 ---
    if [ "$LOGITS_TEST" = "true" ]; then
        run_logits_comparison \
            "${EXPR_PATH}/HF" \
            "${CHECKPOINT_PATH}" \
            "${DEFAULT_LOGITS_PROMPT}" \
            "${MEGATRON_LM_DIR}" \
            "${PDM_DIR}" \
            "${OLD_MEGATRON}" \
            "${LOGITS_TEST_REF}" \
            "${TOKENIZER}"
    fi

elif [ "$MODEL_TYPE" = "apertus" ]; then
    echo "Using Apertus conversion pipeline..."

    APERTUS_ARGS=("${CHECKPOINT_PATH}" --output-dir "${EXPR_PATH}")
    if [ -n "$TOKENIZER" ]; then
        APERTUS_ARGS+=(--tokenizer "${TOKENIZER}")
    fi
    if [ "$OLD_MEGATRON" = "true" ]; then
        APERTUS_ARGS+=(--old-megatron)
    fi
    if [ "$LOGITS_TEST" = "true" ]; then
        APERTUS_ARGS+=(--logits-test)
    fi
    if [ -n "$LOGITS_TEST_REF" ]; then
        APERTUS_ARGS+=(--logits-test-ref "${LOGITS_TEST_REF}")
    fi

    bash "${PDM_DIR}/scripts/swissai_megatron/convert_apertus.sh" "${APERTUS_ARGS[@]}"

    if [ $? -ne 0 ]; then
       echo "Apertus model conversion failed"
       exit 1
    fi
fi

echo ""
echo "Conversion completed successfully!"
