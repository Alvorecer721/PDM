#!/bin/bash
#SBATCH --account=infra01
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/eval/lm-eval_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/eval/lm-eval_%A.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=lm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00

set -e  # Exit immediately if any command fails
set -x  # Print commands for debugging

# Default values
DEFAULT_TOKENIZER="meta-llama/Llama-3.2-3B"
DEFAULT_TASKS="hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"
DEFAULT_INSTRUCT_TASKS="mmlu,gsm8k,ifeval,truthfulqa_mc2,bbh,hellaswag" # mbpp executes code, so not run for now
DEFAULT_INSTRUCT_TOKENIZER="/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"
DEFAULT_BATCH_SIZE="16"
DEFAULT_MAX_LENGTH=""
DEFAULT_APPLY_CHAT_TEMPLATE="false"
DEFAULT_OFFLINE_DATASETS="true"

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
    echo "  --tokenizer TOKENIZER         Tokenizer to use (default: ${DEFAULT_TOKENIZER})"
    echo "  --tasks TASKS                 Comma-separated list of tasks (default: ${DEFAULT_TASKS})"
    echo "  --instruct                    Use instruct benchmarks and apply chat template (sets tasks to: ${DEFAULT_INSTRUCT_TASKS})"
    echo "  --batch-size SIZE             Batch size for evaluation (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --max-length LENGTH           Maximum sequence length (default: no limit)"
    echo "  --apply-chat-template         Apply chat template to inputs (default: ${DEFAULT_APPLY_CHAT_TEMPLATE})"
    echo "  --no-offline-datasets         Disable offline mode for HF datasets (default: offline mode enabled)"
    exit 1
fi

EXPR_PATH="$1"
shift

# Initialize with defaults
TOKENIZER="${DEFAULT_TOKENIZER}"
TASKS="${DEFAULT_TASKS}"
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
MAX_LENGTH="${DEFAULT_MAX_LENGTH}"
APPLY_CHAT_TEMPLATE="${DEFAULT_APPLY_CHAT_TEMPLATE}"
OFFLINE_DATASETS="${DEFAULT_OFFLINE_DATASETS}"
INSTRUCT_MODE="false"
TOKENIZER_EXPLICITLY_SET="false"

# Detect if EXPR_PATH is a HuggingFace identifier or a local path
IS_HF_IDENTIFIER="false"
if [[ ! "$EXPR_PATH" =~ ^[/.] ]] && [[ ! -d "$EXPR_PATH" ]]; then
    IS_HF_IDENTIFIER="true"
    TOKENIZER="Inferred from HF model"
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "LM Evaluation Job Configuration"
echo "========================================"
echo "Experiment path:      ${EXPR_PATH}"
echo "Instruct mode:        ${INSTRUCT_MODE}"
echo "Tokenizer:            ${TOKENIZER}"
echo "Tasks:                ${TASKS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Max length:           ${MAX_LENGTH:-no limit}"
echo "Apply chat template:  ${APPLY_CHAT_TEMPLATE}"
echo "Offline datasets:     ${OFFLINE_DATASETS}"
echo "========================================"
echo ""

# Create log directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/$USER/PDM/log/eval

# Build arguments for the evaluation script
EVAL_ARGS="$EXPR_PATH"
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

# Execute the script inside the compute node using srun
echo "Executing conversion and evaluation script on compute node..."
srun /iopsstor/scratch/cscs/$USER/PDM/scripts/swissai_megatron/convert-and-lm-eval.sh $EVAL_ARGS

echo "Job completed"