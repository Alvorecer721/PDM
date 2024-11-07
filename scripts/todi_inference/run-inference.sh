#!/bin/bash

K=5
H=13
GBS=120
NUM_EPOCHS=93

# Set checkpoint interval and last step (similar to your first script)
STEP_INTERVAL=300
LAST_STEP=$((75 * NUM_EPOCHS))  # 75: num_train_samples / global_batch_size

# Get number of GPUs per node 
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Set your variables
LLAMA_SIZE="1.5B"
DATA_PATH="/users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/data/gutenberg_en_8k_token.jsonl"
LLAMA_CONFIG="/users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/PDM/config/llama3_1.5B_config.json"

# goldfish loss
EXPERIMENT="llama_${LLAMA_SIZE}_Goldfish_K_${K}_H_${H}_GBS_${GBS}_EPOCH_${NUM_EPOCHS}"
HF_MODEL_PATH="/users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/${LLAMA_SIZE}/${EXPERIMENT}/results/NeMo2HF"

# standard loss
# EXPERIMENT="llama_1.5B_Standard_GBS_${GBS}_EPOCH_${NUM_EPOCHS}"
# HF_MODEL_PATH="/store/a06/.NeMo/Goldfish_Llama3//goldfish_ckpts/1b/standard"

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Calculate total number of steps to process
total_steps=$(( (LAST_STEP + STEP_INTERVAL - 1) / STEP_INTERVAL ))
current_step=0

# Loop through steps at intervals
for step in $(seq $STEP_INTERVAL $STEP_INTERVAL $LAST_STEP); do
    current_step=$((current_step + 1))
    consumed=$((step * GBS))
    checkpoint_name="step=${step}-consumed=${consumed}"

    INFERENCE_DIR="/users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/PDM/inference/${EXPERIMENT}/step=${step}-consumed=${consumed}"

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

    python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use-env /users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/PDM/distributed_inference.py \
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

# Handle the last step if it's not a multiple of STEP_INTERVAL
if (( LAST_STEP % STEP_INTERVAL != 0 )); then
    current_step=$((current_step + 1))
    consumed=$((LAST_STEP * GBS))
    checkpoint_name="step=${LAST_STEP}-consumed=${consumed}"
    
    INFERENCE_DIR="/users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/PDM/inference/${EXPERIMENT}/step=${LAST_STEP}-consumed=${consumed}"

    if [ -f "${HF_MODEL_PATH}/${checkpoint_name}.bin" ] && [ ! -d "${INFERENCE_DIR}" ]; then
        echo "Processing final checkpoint ${checkpoint_name}"
        
        python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use-env \
            /users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/PDM/distributed_inference.py \
            --data-path "${DATA_PATH}" \
            --step ${LAST_STEP} \
            --consumed ${consumed} \
            --llama-size "${LLAMA_SIZE}" \
            --llama-config "${LLAMA_CONFIG}" \
            --prefix-length 500 \
            --suffix-length 500 \
            --seq-offset 0 \
            --batch-size 500 \
            --experiment "${EXPERIMENT}" \
            --hf-model-path "${HF_MODEL_PATH}"
    fi
fi