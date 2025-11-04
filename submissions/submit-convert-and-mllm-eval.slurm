#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/eval/mllm-eval_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/eval/mllm-eval_%A.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=mllm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging

# Default values
DEFAULT_MODEL="llama_emu3"
DEFAULT_TASKS="ai2d, mmmu_val, pope, vqav2, mme, gqa, mmstar, ocrbench, seedbench, textvqa, mathvista_testmini, chartqa"
DEFAULT_BATCH_SIZE="1"
DEFAULT_MAX_LENGTH=""
DEFAULT_OFFLINE_DATASETS="true"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Error: No experiment path provided!"
    echo "Usage: sbatch $0 /path/to/experiment/directory [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model MODEL                 MLLM model type (default: ${DEFAULT_MODEL})"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --batch-size SIZE             Batch size for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit)"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL="${DEFAULT_MODEL}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "MLLM Evaluation Job Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "Model:                ${MODEL}"
echo "Tasks:                ${TASKS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "========================================"
echo ""

# Create log directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/$USER/PDM/log/eval

# Build arguments for the evaluation script
EVAL_ARGS="$EXPR_PATH"
if [ "$MODEL" != "${DEFAULT_MODEL}" ]; then
    EVAL_ARGS="$EVAL_ARGS --model $MODEL"
fi
if [ "$TASKS" != "${DEFAULT_TASKS}" ]; then
    EVAL_ARGS="$EVAL_ARGS --tasks $TASKS"
fi
if [ "$BATCH_SIZE" != "${DEFAULT_BATCH_SIZE}" ]; then
    EVAL_ARGS="$EVAL_ARGS --batch-size $BATCH_SIZE"
fi
if [ "$NUM_PROCESSES" != "${DEFAULT_NUM_PROCESSES}" ]; then
    EVAL_ARGS="$EVAL_ARGS --num-processes $NUM_PROCESSES"
fi
if [ -n "$MAX_LENGTH" ]; then
    EVAL_ARGS="$EVAL_ARGS --max-length $MAX_LENGTH"
fi  
if [ "$OFFLINE_DATASETS" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --no-offline-datasets"
fi

# Execute the script inside the compute node using srun
echo "Executing conversion and evaluation script on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert-and-mllm-eval.sh $EVAL_ARGS

echo "Job completed"