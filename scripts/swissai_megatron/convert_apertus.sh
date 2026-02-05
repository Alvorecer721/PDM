#!/bin/bash

# Converts a Megatron checkpoint to HF Apertus format using the Megatron-LM converter
# 1. Converts torch-dist to torch format (if not already done)
# 2. Converts torch to HF Apertus format using the swissai_hf saver
#
# Path arguments:
#   checkpoint_path  - Source torch-dist checkpoint (required)
#   output_dir       - Output directory for converted models (default: parent of checkpoint_path)
#
# Usage: convert_apertus.sh <checkpoint_path> [--output-dir DIR] [--tokenizer TOKENIZER_PATH]
#
# NOTE: The intermediate torch checkpoint will preserve the same tensor parallelism (TP)
#       as the source torch-dist checkpoint. The script auto-detects TP from the checkpoint
#       and launches with the appropriate number of GPUs. The final HF checkpoint is always
#       consolidated (TP=1) since the loader_core handles TP merging during HF conversion.
#
# ATTENTION:
# This script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

# Default tokenizer for Apertus models
DEFAULT_TOKENIZER="alehc/swissai-tokenizer"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "ERROR: Please provide checkpoint path"
    echo "Usage: $0 /path/to/checkpoint [--output-dir DIR] [--tokenizer TOKENIZER_PATH]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path          Direct path to the torch distributed checkpoint"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR         Directory for output (torch/ and HF/ subdirs)"
    echo "                           (default: parent directory of checkpoint_path)"
    echo "  --tokenizer TOKENIZER    HuggingFace tokenizer path or identifier"
    echo "                           (default: ${DEFAULT_TOKENIZER})"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/experiment/checkpoints/3B"
    echo "  $0 /path/to/checkpoint --output-dir /path/to/output"
    exit 1
fi

CHECKPOINT_PATH="$1"
shift

# Initialize with defaults
TOKENIZER="${DEFAULT_TOKENIZER}"
OUTPUT_DIR=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
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

# If output-dir not specified, use parent directory of checkpoint
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$(dirname "${CHECKPOINT_PATH}")"
fi

# Validate checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: ${CHECKPOINT_PATH}"
    exit 1
fi

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
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
    # Compare versions
    if python -c "from packaging import version; exit(0 if version.parse('${current_transformers_version}') >= version.parse('${required_transformers_version}') else 1)"; then
        echo "transformers version ${current_transformers_version} >= ${required_transformers_version} ✓"
    else
        echo "transformers version ${current_transformers_version} < ${required_transformers_version}, upgrading..."
        pip install -U "transformers>=4.56,<5.0.0"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to upgrade transformers to >=${required_transformers_version}"
            exit 1
        fi
    fi
fi

CHECKPOINT_NAME=$(basename ${CHECKPOINT_PATH})

echo "======================================================="
echo "convert_apertus.sh - Apertus Conversion Configuration"
echo "======================================================="
echo "Checkpoint path:  ${CHECKPOINT_PATH}"
echo "Checkpoint name:  ${CHECKPOINT_NAME}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Tokenizer:        ${TOKENIZER}"
echo "Megatron-LM dir:  ${MEGATRON_LM_DIR}"
echo "======================================================="
echo ""

# Check if torch directory already exists, skip conversion if it does
if [ -d "${OUTPUT_DIR}/torch" ]; then
    echo "Torch directory already exists, skipping distributed to torch conversion..."
else
    # Detect TP from checkpoint to determine number of GPUs needed
    # Look for the latest checkpoint iteration
    LATEST_ITER=$(cat "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
    if [ -n "$LATEST_ITER" ]; then
        ITER_DIR="${CHECKPOINT_PATH}/iter_$(printf '%07d' $LATEST_ITER)"
    else
        # Try to find any iter directory
        ITER_DIR=$(find "${CHECKPOINT_PATH}" -maxdepth 1 -type d -name "iter_*" | head -1)
    fi

    if [ -z "$ITER_DIR" ] || [ ! -d "$ITER_DIR" ]; then
        echo "ERROR: Could not find checkpoint iteration directory in ${CHECKPOINT_PATH}"
        exit 1
    fi

    # Extract TP from checkpoint args stored in common.pt
    COMMON_PT="${ITER_DIR}/common.pt"
    if [ -f "$COMMON_PT" ]; then
        TP_SIZE=$(python3 -c "
import torch
d = torch.load('${COMMON_PT}', weights_only=False, map_location='cpu')
args = d.get('args')
if args:
    print(getattr(args, 'tensor_model_parallel_size', 1))
else:
    print(1)
" 2>/dev/null || echo "1")
    else
        echo "WARNING: common.pt not found, defaulting to TP=1"
        TP_SIZE=1
    fi

    echo "Detected tensor_model_parallel_size=${TP_SIZE} from checkpoint"

    # Run the torch distributed to torch conversion with correct number of GPUs
    echo "Converting torch distributed to torch (using ${TP_SIZE} GPUs)..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=${TP_SIZE} \
        ${PDM_DIR}/src/convert/convert_torch_dist_to_torch.py \
        --bf16 \
        --load "${CHECKPOINT_PATH}" \
        --ckpt-convert-save "${OUTPUT_DIR}"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
        echo "Torch distributed to torch conversion failed"
        exit 1
    fi
    echo "Torch conversion completed successfully"
fi

# Check if HF directory already exists
if [ -d "${OUTPUT_DIR}/HF" ]; then
    echo "HF directory already exists, skipping torch to HF conversion..."
    echo "Delete ${OUTPUT_DIR}/HF if you want to reconvert"
    exit 0
fi

# Convert torch to HF Apertus format using Megatron-LM converter
echo ""
echo "Converting torch checkpoint to HF Apertus format..."
echo "Using saver: swissai_hf (outputs Apertus format)"

cd ${MEGATRON_LM_DIR}

python tools/checkpoint/convert.py \
    --model-type=GPT \
    --loader=core \
    --saver=swissai_hf \
    --load-dir="${OUTPUT_DIR}/torch" \
    --save-dir="${OUTPUT_DIR}/HF" \
    --hf-tokenizer="${TOKENIZER}"

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion to HF Apertus format failed"
   exit 1
fi

echo ""
echo "========================================"
echo "Apertus Conversion Completed Successfully"
echo "========================================"
echo "HF Apertus model saved to: ${OUTPUT_DIR}/HF"
echo ""
echo "You can now load the model with:"
echo "  from transformers import AutoModelForCausalLM"
echo "  model = AutoModelForCausalLM.from_pretrained('${OUTPUT_DIR}/HF')"
echo "========================================"
