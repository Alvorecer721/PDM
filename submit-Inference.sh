#!/bin/bash

K=21
H=13
GBS=120
NUM_EPOCHS=79

# Get number of GPUs per node 
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Set your variables
LLAMA_SIZE="1.5B"
DATA_PATH="/mloscratch/homes/yixuan/gutenberg_en_8k_token.jsonl"
LLAMA_CONFIG="/mloscratch/homes/yixuan/PDM/config/llama3_1.5B_config.json"

# goldfish loss
EXPERIMENT="llama_1.5B_Goldfish_K_${K}_H_${H}_GBS_${GBS}_EPOCH_${NUM_EPOCHS}"
HF_MODEL_PATH="/mloscratch/homes/yixuan/goldfish_ckpts/1b/k_${K}_h_${H}"

# standard loss
# EXPERIMENT="llama_1.5B_Standard_GBS_${GBS}_EPOCH_${NUM_EPOCHS}"
# HF_MODEL_PATH="/mloscratch/homes/yixuan/goldfish_ckpts/1b/standard"

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

for checkpoint in ${HF_MODEL_PATH}/*.bin; do
    checkpoint_name=$(basename ${checkpoint} .bin)
    
    # Using parameter expansion to extract values
    step=${checkpoint_name#*step=}    # Remove everything before "step="
    step=${step%%-*}                  # Remove everything after first "-"
    
    consumed=${checkpoint_name#*consumed=}  # Remove everything before "consumed="
    consumed=${consumed%.*}                 # Remove ".bin" extension

    INFERENCE_DIR="/mloscratch/homes/yixuan/PDM/inference/${EXPERIMENT}/step=${step}-consumed=${consumed}"

    # Check if inference output already exists
    if [ -d "${INFERENCE_DIR}" ]; then
        echo "Skipping ${checkpoint_name} - inference exists in ${INFERENCE_DIR}"
        echo "----------------------------------------"
        continue
    fi

    # Run the inference script
    echo "Starting inference with:"
    echo "Step: ${step}"
    echo "Consumed: ${consumed}"
    echo "Experiment: ${EXPERIMENT}"
    echo "Using ${NUM_GPUS} GPUs per node"

    python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use-env distributed_inference.py \
        --data-path "${DATA_PATH}" \
        --step ${step} \
        --consumed ${consumed} \
        --llama-size "${LLAMA_SIZE}" \
        --llama-config "${LLAMA_CONFIG}" \
        --prefix-length 500 \
        --suffix-length 500 \
        --seq-offset 0 \
        --batch-size 500 \
        --experiment "${EXPERIMENT}" \
        --hf-model-path "${HF_MODEL_PATH}"

    echo "Completed inference for ${checkpoint_name}"
    echo "----------------------------------------"
done