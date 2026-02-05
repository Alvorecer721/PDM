#!/bin/bash

#
# Script that runs model conversion and runs evaluation of the model with lm-eval
# For conversion of apertus checkpoints it might be necessary to use a nemo container - checkpoints and conversions must be done with same container
# Ex If training was run with the latest nemo container, the conversion must be run with the same (ngc-container might not work)
#

#SBATCH --account=infra01
# Environment controlled via command-line: --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --gres=gpu:4
#SBATCH --job-name=lm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging
ulimit -c 0 # no core dumps

# Default values
DEFAULT_MODEL_TYPE="llama3"
DEFAULT_TOKENIZER="meta-llama/Llama-3.2-3B"
DEFAULT_TASKS="hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"
DEFAULT_INSTRUCT_TASKS="mmlu,gsm8k,ifeval,truthfulqa_mc2,bbh,hellaswag" # mbpp executes code, so not run for now
DEFAULT_INSTRUCT_TOKENIZER="/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"
DEFAULT_BATCH_SIZE="12"
DEFAULT_MAX_LENGTH=""
DEFAULT_APPLY_CHAT_TEMPLATE="false"
DEFAULT_OFFLINE_DATASETS="true"
DEFAULT_LOG_SUBDIR="log/eval"
DEFAULT_GROUP_NAME="default"

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo "Error: No experiment path or HuggingFace model identifier provided!"
    echo "Usage: sbatch $0 <experiment_path|hf_model_id> [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 /path/to/megatron/experiment"
    echo "  sbatch $0 meta-llama/Llama-3.2-3B --instruct"
    echo ""
    echo "Options:"
    echo "  --model-type TYPE             Model architecture type (default: ${DEFAULT_MODEL_TYPE})"
    echo "                                Options: llama3, apertus"
    echo "  --tokenizer TOKENIZER         Tokenizer to use (default: model-type specific)"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --instruct                    Use instruct benchmarks and apply chat template (sets tasks to: ${DEFAULT_INSTRUCT_TASKS})"
    echo "  --batch-size SIZE             Batch size for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit)"
    echo "  --apply-chat-template         Apply chat template to inputs (default: ${DEFAULT_APPLY_CHAT_TEMPLATE})"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    echo "  --no-convert                  Skip model conversion (expects HF directory to exist)"
    echo "  --log-subdir SUBDIR           Log subdirectory relative to PDM root (default: ${DEFAULT_LOG_SUBDIR})"
    echo "  --group-name NAME             Group name for organizing logs (default: ${DEFAULT_GROUP_NAME})"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
MODEL_TYPE="${DEFAULT_MODEL_TYPE}"
TOKENIZER="${DEFAULT_TOKENIZER}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"
INSTRUCT_MODE="false"
TOKENIZER_EXPLICITLY_SET="false"
NO_CONVERT="false"
LOG_SUBDIR="${DEFAULT_LOG_SUBDIR}"
GROUP_NAME="${DEFAULT_GROUP_NAME}"

# Detect if EXPR_PATH is a HuggingFace identifier or a local path
IS_HF_IDENTIFIER="false"
if [[ ! "$EXPR_PATH" =~ ^[/.] ]] && [[ ! -d "$EXPR_PATH" ]]; then
    IS_HF_IDENTIFIER="true"
    TOKENIZER="Inferred from HF model"
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            TOKENIZER_EXPLICITLY_SET="true"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --instruct)
            INSTRUCT_MODE="true"
            TASKS="${DEFAULT_INSTRUCT_TASKS}"
            APPLY_CHAT_TEMPLATE="true"
            # only set default instruct tokenizer, if not hf identifier
            if [ "$IS_HF_IDENTIFIER" = "false" ]; then
              TOKENIZER="${DEFAULT_INSTRUCT_TOKENIZER}"
            fi
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --apply-chat-template)
            APPLY_CHAT_TEMPLATE="true"
            shift
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set model-type specific defaults if tokenizer not explicitly set and not in instruct mode
if [ "$TOKENIZER_EXPLICITLY_SET" = "false" ] && [ "$INSTRUCT_MODE" = "false" ] && [ "$IS_HF_IDENTIFIER" = "false" ]; then
    if [ "$MODEL_TYPE" = "apertus" ]; then
        TOKENIZER="alehc/swissai-tokenizer"
    else
        TOKENIZER="meta-llama/Llama-3.2-3B"
    fi
fi

# Setup log directory and redirect output
LOG_DIR="/iopsstor/scratch/cscs/$USER/PDM/${LOG_SUBDIR}"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/lm-eval_${GROUP_NAME}_${SLURM_JOB_ID}.out" 2>&1

# Print configuration
echo "========================================"
echo "LM Evaluation Job Configuration"
echo "========================================"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Log Directory:        ${LOG_DIR}"
echo "Group Name:           ${GROUP_NAME}"
echo "Experiment path:      ${EXPR_PATH}"
echo "Instruct mode:        ${INSTRUCT_MODE}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Tasks:                ${TASKS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Apply chat template:  ${APPLY_CHAT_TEMPLATE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "No convert:           ${NO_CONVERT}"
echo "========================================"
echo ""

# Build arguments for the evaluation script
EVAL_ARGS="$EXPR_PATH"
# Add model type
if [ "$MODEL_TYPE" != "${DEFAULT_MODEL_TYPE}" ]; then
    EVAL_ARGS="$EVAL_ARGS --model-type $MODEL_TYPE"
fi
# Only pass tokenizer if explicitly set OR if it's not a HF identifier
if [ "$TOKENIZER_EXPLICITLY_SET" = "true" ] || [ "$IS_HF_IDENTIFIER" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --tokenizer $TOKENIZER"
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
if [ "$APPLY_CHAT_TEMPLATE" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --apply-chat-template"
fi
if [ "$OFFLINE_DATASETS" = "false" ]; then
    EVAL_ARGS="$EVAL_ARGS --no-offline-datasets"
fi
if [ "$NO_CONVERT" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --no-convert"
fi

# Execute the script inside the compute node using srun
echo "Executing conversion and evaluation script on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert-and-lm-eval.sh $EVAL_ARGS

echo "Job completed"