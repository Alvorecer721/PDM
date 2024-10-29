#!/bin/bash

# Set your variables
STEP=75
CONSUMED=9000
LLAMA_SIZE="1.5B"
HF_TOKEN="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"
DATA_PATH="/mloscratch/homes/yixuan/PDM"
LLAMA_CONFIG="/Users/xuyixuan/Downloads/Project/PDM/PDM/config/llama3_1.5B_config.json"
EXPERIMENT="llama_1.5B_Goldfish_K_21_H_13_GBS_120_EPOCH_79"
HF_MODEL_PATH="/mloscratch/homes/yixuan/"


# Run the inference script
echo "Starting inference with:"
echo "Step: ${STEP}"
echo "Consumed: ${CONSUMED}"
echo "Experiment: ${EXPERIMENT}"

python distributed_inference.py \
    --data-path "${DATA_PATH}" \
    --hf-token "${HF_TOKEN}" \
    --step ${STEP} \
    --consumed ${CONSUMED} \
    --llama-size "${LLAMA_SIZE}" \
    --llama-config "${LLAMA_CONFIG}" \
    --prefix-length 500 \
    --suffix-length 500 \
    --seq-offset 0 \
    --batch-size 500 \
    --experiment "${EXPERIMENT}" \
    --hf-model-path "${HF_MODEL_PATH}"