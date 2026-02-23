#!/bin/bash
#SBATCH --account=a-infra01
# Environment controlled via command-line: --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --gres=gpu:4
#SBATCH --job-name=mllm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --reservation=PA-2338-RL

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging
ulimit -c 0 # no core dumps

# Default values
DEFAULT_MODEL_TYPE="llama3"
DEFAULT_MODEL="llama_emu3"
DEFAULT_TASKS="pope,gqa,vqav2_val,mmmu_val_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"
DEFAULT_BATCH_SIZE="1"
DEFAULT_MAX_LENGTH=""
DEFAULT_OFFLINE_DATASETS="true"
DEFAULT_LOG_SUBDIR="log/eval"
DEFAULT_GROUP_NAME="default"
DEFAULT_EMU_MIN_PIXELS=""
DEFAULT_EMU_MAX_PIXELS=""
WANDB_ENABLED="true"
WANDB_PROJECT=""
WANDB_RUN_ID=""

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Error: No experiment path provided!"
    echo "Usage: sbatch $0 /path/to/experiment/directory [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE             Model architecture type (default: ${DEFAULT_MODEL_TYPE})"
    echo "                                Options: llama3, apertus"
    echo "  --tokenizer TOKENIZER         Tokenizer to use (optional, model-type specific default)"
    echo "  --model MODEL                 MLLM model type (default: ${DEFAULT_MODEL})"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --batch-size SIZE             Batch size for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit)"
    echo "  --emu-min-pixels PIXELS      Minimum pixels for EMU vision encoder (e.g., 262144 for 512*512)"
    echo "  --emu-max-pixels PIXELS      Maximum pixels for EMU vision encoder (e.g., 1048576 for 1024*1024)"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    echo "  --no-convert                  Skip model conversion (expects HF directory to exist)"
    echo "  --log-subdir SUBDIR           Log subdirectory relative to PDM root (default: ${DEFAULT_LOG_SUBDIR})"
    echo "  --group-name NAME             Group name for organizing logs (default: ${DEFAULT_GROUP_NAME})"
    echo "  --no-wandb                    Disable wandb logging (default: enabled)"
    echo "  --wandb-project PROJECT       Wandb project name (default: lmms-eval)"
    echo "  --debug                       Add debug params to llmms eval run, prints 1st 5 samples with debug info"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL_TYPE="${DEFAULT_MODEL_TYPE}"
TOKENIZER=""
MODEL="${DEFAULT_MODEL}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"
NO_CONVERT="false"
TOKENIZER_EXPLICITLY_SET="false"
LOG_SUBDIR="${DEFAULT_LOG_SUBDIR}"
GROUP_NAME="${DEFAULT_GROUP_NAME}"
EMU_MIN_PIXELS="${DEFAULT_EMU_MIN_PIXELS}"
EMU_MAX_PIXELS="${DEFAULT_EMU_MAX_PIXELS}"
DEBUG=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        --tokenizer)
            TOKENIZER="$2"
            TOKENIZER_EXPLICITLY_SET="true"
            shift 2
            ;;
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
        --emu-min-pixels)
            EMU_MIN_PIXELS="$2"
            shift 2
            ;;
        --emu-max-pixels)
            EMU_MAX_PIXELS="$2"
            shift 2
            ;;
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        --no-convert)
            NO_CONVERT="true"
            shift
            ;;
        --log-subdir)
            LOG_SUBDIR="$2"
            shift 2
            ;;
        --group-name)
            GROUP_NAME="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_ENABLED="false"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-run-id)
            WANDB_RUN_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set tokenizer to model path by default if not explicitly set
if [ "$TOKENIZER_EXPLICITLY_SET" = "false" ]; then
    TOKENIZER="$EXPR_PATH"
fi

# Setup log directory and redirect output
LOG_DIR="/iopsstor/scratch/cscs/$USER/PDM/${LOG_SUBDIR}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/mllm-eval_${GROUP_NAME}_${SLURM_JOB_ID}.out" 2>&1

# Print configuration
echo "========================================"
echo "MLLM Evaluation Job Configuration"
echo "========================================"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Log Directory:        ${LOG_DIR}"
echo "Group Name:           ${GROUP_NAME}"
echo "Experiment path:      ${EXPR_PATH}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Model:                ${MODEL}"
echo "Tasks:                ${TASKS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "No convert:           ${NO_CONVERT}"
echo "Wandb enabled:        ${WANDB_ENABLED}"
if [ -n "$WANDB_PROJECT" ]; then
    echo "Wandb project:        ${WANDB_PROJECT}"
fi
echo "========================================"
echo ""

# Build arguments for the evaluation script
EVAL_ARGS="$EXPR_PATH"
if [ "$MODEL_TYPE" != "${DEFAULT_MODEL_TYPE}" ]; then
    EVAL_ARGS="$EVAL_ARGS --model-type $MODEL_TYPE"
fi
if [ -n "$TOKENIZER" ]; then
    EVAL_ARGS="$EVAL_ARGS --tokenizer $TOKENIZER"
fi
if [ "$MODEL" != "${DEFAULT_MODEL}" ]; then
    EVAL_ARGS="$EVAL_ARGS --model $MODEL"
fi
if [ "$TASKS" != "${DEFAULT_TASKS}" ]; then
    EVAL_ARGS="$EVAL_ARGS --tasks $TASKS"
fi
if [ "$BATCH_SIZE" != "${DEFAULT_BATCH_SIZE}" ]; then
    EVAL_ARGS="$EVAL_ARGS --batch-size $BATCH_SIZE"
fi
if [ -n "$MAX_LENGTH" ]; then
    EVAL_ARGS="$EVAL_ARGS --max-length $MAX_LENGTH"
fi
if [ -n "$EMU_MIN_PIXELS" ]; then
    EVAL_ARGS="$EVAL_ARGS --emu-min-pixels $EMU_MIN_PIXELS"
fi
if [ -n "$EMU_MAX_PIXELS" ]; then
    EVAL_ARGS="$EVAL_ARGS --emu-max-pixels $EMU_MAX_PIXELS"
fi
if [ "$OFFLINE_DATASETS" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --no-offline-datasets"
fi
if [ "$NO_CONVERT" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --no-convert"
fi
if [ "$DEBUG" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --debug"
fi
if [ "$WANDB_ENABLED" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --no-wandb"
fi
if [ -n "$WANDB_PROJECT" ]; then
    EVAL_ARGS="$EVAL_ARGS --wandb-project $WANDB_PROJECT"
fi
if [ -n "$GROUP_NAME" ]; then
    EVAL_ARGS="$EVAL_ARGS --group-name $GROUP_NAME"
fi
if [ -n "$WANDB_RUN_ID" ]; then
    EVAL_ARGS="$EVAL_ARGS --wandb-run-id $WANDB_RUN_ID"
fi

# Execute the script inside the compute node using srun
echo "Executing conversion and evaluation script on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert-and-mllm-eval.sh $EVAL_ARGS

echo "Job completed"