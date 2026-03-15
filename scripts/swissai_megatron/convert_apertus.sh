#!/bin/bash
set -e  # Exit on error

# Converts a Megatron checkpoint to HF Apertus format using the Megatron-LM converter
# 1. Converts torch-dist to torch format (if not already done)
# 2. Converts torch to HF Apertus format using the swissai_hf saver
#
# By default uses the new Megatron API (model_provider + gpt_builder).
# Pass --old-megatron to use the legacy pretrain_gpt model_provider (deprecated).
#
# Path arguments:
#   checkpoint_path  - Source torch-dist checkpoint (required)
#
# Usage: convert_apertus.sh <checkpoint_path> [OPTIONS]
#
# NOTE: The intermediate torch checkpoint will preserve the same tensor parallelism (TP)
#       as the source torch-dist checkpoint. The script auto-detects TP from the checkpoint
#       and launches with the appropriate number of GPUs. The final HF checkpoint is always
#       consolidated (TP=1) since the loader_core handles TP merging during HF conversion.
#
# ATTENTION:
# This script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Source shared utilities
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Default tokenizer for Apertus models
DEFAULT_TOKENIZER="/users/rkreft/MLLM-infra01-folder/tokenizer/apertus_emu3.5_wavtok"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide checkpoint path"
    echo "Usage: $0 /path/to/checkpoint [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path              Direct path to the torch distributed checkpoint"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR             Directory for output (torch/ and HF/ subdirs)"
    echo "                               (default: parent directory of checkpoint_path)"
    echo "  --torch-output-dir DIR       Explicit output directory for torch checkpoint"
    echo "                               (default: <output-dir>/torch)"
    echo "                               NOTE: Must end with '/torch' (Megatron creates this subdir)"
    echo "  --hf-output-dir DIR          Explicit output directory for HF checkpoint"
    echo "                               (default: <output-dir>/HF)"
    echo "  --megatron-dir DIR           Path to Megatron-LM repo"
    echo "                               (default: /iopsstor/scratch/cscs/\$USER/Megatron-LM)"
    echo "  --tokenizer TOKENIZER        HuggingFace tokenizer path or identifier"
    echo "                               (default: ${DEFAULT_TOKENIZER})"
    echo "  --old-megatron               Use legacy pretrain_gpt model_provider (deprecated)"
    echo "  --logits-test                Run logits comparison after HF conversion"
    echo "  --logits-test-ref SPEC       Reference checkpoint for logits comparison"
    echo "                               (e.g. hf:/path/to/ref or meg:/path/to/ckpt)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/experiment/checkpoints/3B"
    echo "  $0 /path/to/checkpoint --output-dir /path/to/output"
    echo "  $0 /path/to/checkpoint --torch-output-dir /tmp/output/torch --hf-output-dir /results/HF"
    echo "  $0 /path/to/checkpoint --old-megatron  # use legacy Megatron API"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift

# Initialize with defaults
TOKENIZER="${DEFAULT_TOKENIZER}"
OUTPUT_DIR=""
TORCH_OUTPUT_DIR=""
HF_OUTPUT_DIR=""
MEGATRON_LM_DIR=""
OLD_MEGATRON="false"
LOGITS_TEST="false"
LOGITS_TEST_REF=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --torch-output-dir)
            TORCH_OUTPUT_DIR="$2"
            shift 2
            ;;
        --hf-output-dir)
            HF_OUTPUT_DIR="$2"
            shift 2
            ;;
        --megatron-dir)
            MEGATRON_LM_DIR="$2"
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

# If output-dir not specified, use parent directory of checkpoint
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(dirname "${CHECKPOINT_PATH}")"
fi

# Set torch and HF output dirs from explicit flags, falling back to output-dir subdirs
if [ -z "$TORCH_OUTPUT_DIR" ]; then
    TORCH_OUTPUT_DIR="${OUTPUT_DIR}/torch"
fi
if [ -z "$HF_OUTPUT_DIR" ]; then
    HF_OUTPUT_DIR="${OUTPUT_DIR}/HF"
fi

# Validate --torch-output-dir ends with /torch (Megatron creates this subdir automatically)
TORCH_BASENAME=$(basename "${TORCH_OUTPUT_DIR}")
if [ "$TORCH_BASENAME" != "torch" ]; then
    echo "ERROR: --torch-output-dir must end with '/torch' because Megatron's"
    echo "       --ckpt-convert-save automatically creates a 'torch/' subdirectory."
    echo "       Got: ${TORCH_OUTPUT_DIR}"
    echo "       Try: $(dirname "${TORCH_OUTPUT_DIR}")/torch"
    exit 1
fi

# Validate checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: ${CHECKPOINT_PATH}"
    exit 1
fi

if [ -z "$MEGATRON_LM_DIR" ]; then
    MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
fi
PDM_DIR=/iopsstor/scratch/cscs/$USER/PDM

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Check transformers version (Apertus was integrated in 4.56.0)
required_transformers_version="4.56.0"
current_transformers_version=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)

if [ -z "$current_transformers_version" ]; then
    echo "transformers not found, installing transformers>=${required_transformers_version}..."
    pip install "transformers>=4.56,<5.0.0"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install transformers>=${required_transformers_version}"
        exit 1
    fi
else
    if python -c "from packaging import version; exit(0 if version.parse('${current_transformers_version}') >= version.parse('${required_transformers_version}') else 1)"; then
        echo "transformers version ${current_transformers_version} >= ${required_transformers_version} OK"
    else
        echo "transformers version ${current_transformers_version} < ${required_transformers_version}, upgrading..."
        pip install -U "transformers>=4.56,<5.0.0"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to upgrade transformers to >=${required_transformers_version}"
            exit 1
        fi
    fi
fi

CHECKPOINT_NAME=$(basename "${CHECKPOINT_PATH}")

echo "============================================================="
echo "convert_apertus.sh - Apertus Conversion Configuration"
echo "============================================================="
echo "Checkpoint path:  ${CHECKPOINT_PATH}"
echo "Checkpoint name:  ${CHECKPOINT_NAME}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Torch output dir: ${TORCH_OUTPUT_DIR}"
echo "HF output dir:    ${HF_OUTPUT_DIR}"
echo "Tokenizer:        ${TOKENIZER}"
echo "Megatron-LM dir:  ${MEGATRON_LM_DIR}"
echo "Old Megatron:     ${OLD_MEGATRON}"
echo "============================================================="
echo ""

# The torch conversion script expects --ckpt-convert-save to be the parent dir
# and creates a "torch" subdir inside it.
TORCH_CONVERT_SAVE_DIR="$(dirname "${TORCH_OUTPUT_DIR}")"

# Check if torch directory already exists, skip conversion if it does
if [ -d "${TORCH_OUTPUT_DIR}" ]; then
    echo "Torch directory already exists at ${TORCH_OUTPUT_DIR}, skipping distributed to torch conversion..."
else
    detect_tp_size "${CHECKPOINT_PATH}"

    # Build conversion command as array for safe quoting
    CONVERT_ARGS=(
        "${PDM_DIR}/src/convert/convert_torch_dist_to_torch.py"
        --bf16
        --load "${CHECKPOINT_PATH}"
        --ckpt-convert-save "${TORCH_CONVERT_SAVE_DIR}"
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
    echo "Torch conversion completed successfully"
fi

# Check if HF directory already exists
if [ -d "${HF_OUTPUT_DIR}" ]; then
    echo "HF directory already exists at ${HF_OUTPUT_DIR}, skipping torch to HF conversion..."
    echo "Delete ${HF_OUTPUT_DIR} if you want to reconvert"
else
    echo ""
    echo "Converting torch checkpoint to HF Apertus format..."
    echo "Using saver: swissai_hf (outputs Apertus format)"

    cd "${MEGATRON_LM_DIR}"

    python tools/checkpoint/convert.py \
        --model-type=GPT \
        --loader=core \
        --saver=swissai_hf \
        --load-dir="${TORCH_OUTPUT_DIR}" \
        --save-dir="${HF_OUTPUT_DIR}" \
        --hf-tokenizer="${TOKENIZER}"

    if [ $? -ne 0 ]; then
       echo "Model conversion to HF Apertus format failed"
       exit 1
    fi
fi

# --- Optional logits comparison ---
if [ "$LOGITS_TEST" = "true" ]; then
    run_logits_comparison \
        "${HF_OUTPUT_DIR}" \
        "${CHECKPOINT_PATH}" \
        "${DEFAULT_LOGITS_PROMPT}" \
        "${MEGATRON_LM_DIR}" \
        "${PDM_DIR}" \
        "${OLD_MEGATRON}" \
        "${LOGITS_TEST_REF}" \
        "${TOKENIZER}"
fi

echo ""
echo "========================================"
echo "Apertus Conversion Completed Successfully"
echo "========================================"
echo "HF Apertus model saved to: ${HF_OUTPUT_DIR}"
echo ""
echo "You can now load the model with:"
echo "  from transformers import AutoModelForCausalLM"
echo "  model = AutoModelForCausalLM.from_pretrained('${HF_OUTPUT_DIR}')"
echo "========================================"
