#!/bin/bash

# Set your variables
STEP=1200
CONSUMED=$((STEP * 120))
LLAMA_SIZE="1.5B"
HF_TOKEN="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"
DATA_PATH="/mloscratch/homes/yixuan/gutenberg_en_8k_token.jsonl"
LLAMA_CONFIG="/mloscratch/homes/yixuan/PDM/config/llama3_1.5B_config.json"
EXPERIMENT="llama_1.5B_Goldfish_K_21_H_13_GBS_120_EPOCH_79"
HF_MODEL_PATH="/mloscratch/homes/yixuan/goldfish_ckpts/1b/k_21_h_13"


# Run the inference script
echo "Starting inference with:"
echo "Step: ${STEP}"
echo "Consumed: ${CONSUMED}"
echo "Experiment: ${EXPERIMENT}"

python -m torch.distributed.launch --nproc_per_node=1 --use-env distributed_inference.py \
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
