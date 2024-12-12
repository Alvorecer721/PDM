#!/bin/bash

# Configuration variables (direct assignment is better)
K=54
H=13
NUM_EPOCHS=76
GBS=120
STEP_INTERVAL=300
LLAMA_SIZE="1.5B"
STEPS_PER_EPOCH=75

# Paths configuration (direct assignment)
BASE_DIR="/users/xyixuan/store/.NeMo/Goldfish_Llama3"
DATA_PATH="${BASE_DIR}/data/gutenberg_en_8k_token.jsonl"
LLAMA_CONFIG="${BASE_DIR}/PDM/config/llama3_1.5B_config.json"
INFERENCE_SCRIPT="${BASE_DIR}/PDM/distributed_inference.py"

# Calculated variables
LAST_STEP=$((STEPS_PER_EPOCH * NUM_EPOCHS))
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Setup experiment name and paths
function setup_experiment() {
    local exp_type=$1
    if [ "$exp_type" = "goldfish" ]; then
        EXPERIMENT="llama_${LLAMA_SIZE}_Goldfish_K_${K}_H_${H}_GBS_${GBS}_EPOCH_${NUM_EPOCHS}"
        # HF_MODEL_PATH="${BASE_DIR}/${LLAMA_SIZE}/${EXPERIMENT}/results/NeMo2HF"
        HF_MODEL_PATH="/iopsstor/scratch/cscs/xyixuan/llama_1.5B_Goldfish_K_54_H_13_GBS_120_EPOCH_76/results/NeMo2HF"
    else
        EXPERIMENT="llama_${LLAMA_SIZE}_Standard_GBS_${GBS}_EPOCH_${NUM_EPOCHS}"
        HF_MODEL_PATH="${BASE_DIR}/goldfish_ckpts/1b/standard"
    fi
}

# Run inference for a specific checkpoint
function run_ckpt() {
    local step=$1
    local consumed=$((step * GBS))
    local checkpoint_name="step=${step}-consumed=${consumed}"
    local inference_dir="${BASE_DIR}/PDM/inference/${EXPERIMENT}/step=${step}-consumed=${consumed}"

    # Check if inference already exists
    if [ -d "${inference_dir}" ]; then
        echo "Skipping ${checkpoint_name} - inference exists in ${inference_dir}"
        return 0
    fi

    echo "Starting inference with:"
    echo "Step: ${step}"
    echo "Consumed: ${consumed}"
    echo "Experiment: ${EXPERIMENT}"
    echo "Using ${NUM_GPUS} GPUs per node"

    python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --use-env ${INFERENCE_SCRIPT} \
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
}

# Process checkpoints at intervals
function inference() {
    local total_steps=$(( (LAST_STEP + STEP_INTERVAL - 1) / STEP_INTERVAL ))
    local current_step=0

    # Process regular intervals
    for step in $(seq ${STEP_INTERVAL} ${STEP_INTERVAL} ${LAST_STEP}); do
        current_step=$((current_step + 1))
        echo "[Progress: ${current_step}/${total_steps}]"
        run_ckpt ${step}
    done

    # Process final step if needed
    if (( LAST_STEP % STEP_INTERVAL != 0 )); then
        current_step=$((current_step + 1))
        echo "[Final step: ${current_step}/${total_steps}]"
        run_ckpt ${LAST_STEP}
    fi
}

# Main execution
function main() {
    # Export required environment variables
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

    # Print configuration summary
    echo "Configuration:"
    echo "- Epochs: ${NUM_EPOCHS}"
    echo "- Steps per epoch: ${STEPS_PER_EPOCH}"
    echo "- Last step: ${LAST_STEP}"
    echo "- Step interval: ${STEP_INTERVAL}"
    echo "----------------------------------------"

    # Setup experiment (goldfish or standard)
    setup_experiment "goldfish"

    # Process all checkpoints
    inference

    echo "All inference tasks completed!"
}

# Execute main function
main