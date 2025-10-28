#!/bin/bash

# Converts a megatron checkpoint 1st to torch-dist and then to a hf checkpoint. Both only if the respective folder
# doesnt exist. Argument: path to Meg-Run folder.
# ATTENTION:
# this script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml

if [ "$#" -ne 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory"
    exit 1
fi

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

EXPR_PATH="$1"
EXPR_NAME=$(basename ${EXPR_PATH})


# Check if torch directory already exists, skip conversion if it does
if [ -d "${EXPR_PATH}/torch" ]; then
    echo "Torch directory already exists, skipping distributed to torch conversion..."
else
    # Run the torch distributed to torch conversion
    echo "Converting torch distributed to torch..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun /iopsstor/scratch/cscs/$USER/PDM/src/convert/convert_torch_dist_to_torch.py \
        --bf16 \
        --load ${EXPR_PATH}/checkpoints/3B \
        --ckpt-convert-save "${EXPR_PATH}"

    # Check if the conversion was successful
    if [ $? -ne 0 ]; then
        echo "Torch distributed to torch conversion failed"
        exit 1
    fi
fi

python /iopsstor/scratch/cscs/$USER/PDM/src/convert/convert_megatron_to_hf.py \
   --experiment-path "${EXPR_PATH}"

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion failed"
   exit 1
fi
