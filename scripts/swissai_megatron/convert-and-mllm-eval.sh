#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory"
    exit 1
fi

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lmms-eval

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

EXPR_PATH="$1"
EXPR_NAME=$(basename ${EXPR_PATH})
RES_PATH="/iopsstor/scratch/cscs/$USER/PDM/results/lmms_eval/${EXPR_NAME}"

# Clear existing results directory if it exists
if [ -d "${RES_PATH}" ]; then
    echo "Clearing existing results directory: ${RES_PATH}"
    rm -rf "${RES_PATH}"/*
fi


if [ ! -d "$EVAL_DIR" ]; then
    echo "Creating lmms-eval directory..."
    git clone https://github.com/nirmiger/lmms-eval.git "$EVAL_DIR"
else
    echo "lmms-eval directory exists, updating repository..."
    cd "$EVAL_DIR"
    git pull
fi

# Create results directory
mkdir -p ${RES_PATH}

# Check if torch directory already exists, skip conversion if it does
if [ -d "${EXPR_PATH}/torch" ]; then
    echo "Torch directory already exists, skipping distributed to torch conversion..."
else
    # Run the torch distributed to torch conversion
    echo "Converting torch distributed to torch..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun /iopsstor/scratch/cscs/$USER/PDM/src/convert/convert_torch_dist_to_torch.py \
        --bf16 \
        --load ${EXPR_PATH}/checkpoints/3B \
        --ckpt-convert-save ${EXPR_PATH}

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
        echo "Torch distributed to torch conversion failed"
        exit 1
    fi
fi

python /iopsstor/scratch/cscs/$USER/PDM/src/convert/convert_megatron_to_hf.py \
   --experiment-path ${EXPR_PATH}

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion failed"
   exit 1
fi

# Install/update lm-eval
echo "📦 Setting up lmms-eval package..."
cd "$EVAL_DIR"
pip uninstall jupyterlab -y  # Uninstall conflicting package
pip install -e .
pip install qwen_vl_utils

# Then run your evaluation command
accelerate launch --num_processes=4 -m lmms_eval \
    --model llama_vision \
    --model_args pretrained=${EXPR_PATH}/HF \
    --tasks ai2d,docvqa_val \
    --batch_size 1 \
    --output_path ${RES_PATH}

# ATTENTION: 
# this script should be run inside computing node with:
# --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml