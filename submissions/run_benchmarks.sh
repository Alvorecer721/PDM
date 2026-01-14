#!/bin/bash

# Orchestrator script for launching multiple benchmarking jobs with task groups
# Supports standalone conversion, LM evaluation, and MLLM evaluation with SLURM dependencies

set -e  # Exit on error

# Color codes for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default task groups
DEFAULT_LM_TASKS='[{"name":"default","tasks":"hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"}]'
DEFAULT_MLLM_TASKS='[{"name":"default","tasks":"ai2d,mmmu_val,pope,vqav2,mme,gqa,mmstar,ocrbench,seedbench,textvqa,mathvista_testmini,chartqa"}]'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage function
usage() {
    echo "Usage: $0 <experiment_path|hf_model_id> [OPTIONS]"
    echo ""
    echo "Orchestrator for launching multiple benchmarking jobs with task groups."
    echo ""
    echo "Arguments:"
    echo "  experiment_path|hf_model_id    Path to Megatron experiment or HuggingFace model identifier"
    echo ""
    echo "Action Flags (at least one required):"
    echo "  --convert                      Launch standalone conversion job"
    echo "  --lm-eval                      Launch LM evaluation jobs"
    echo "  --lmms-eval                    Launch MLLM evaluation jobs"
    echo ""
    echo "Task Group Options:"
    echo "  --lm-task-groups JSON_FILE     JSON file defining LM eval task groups"
    echo "  --mllm-task-groups JSON_FILE   JSON file defining MLLM eval task groups"
    echo ""
    echo "Evaluation Options:"
    echo "  --instruct                     Use instruct mode for LM eval (sets chat template)"
    echo "  --batch-size SIZE              Batch size for evaluation"
    echo "  --tokenizer TOKENIZER          Tokenizer to use (LM eval only)"
    echo "  --model MODEL                  MLLM model type (MLLM eval only)"
    echo ""
    echo "Task Group JSON Format:"
    echo '  [{"name":"reasoning","tasks":"gsm8k,bbh,mmlu"},{"name":"text","tasks":"hellaswag,winogrande"}]'
    echo ""
    echo "Examples:"
    echo "  # Convert only"
    echo "  $0 /path/to/experiment --convert"
    echo ""
    echo "  # Convert and run LM eval with custom task groups"
    echo "  $0 /path/to/experiment --convert --lm-eval --lm-task-groups lm_groups.json --instruct"
    echo ""
    echo "  # Run MLLM eval without conversion (already converted)"
    echo "  $0 /path/to/experiment --lmms-eval --mllm-task-groups mllm_groups.json"
    echo ""
    echo "  # Full pipeline"
    echo "  $0 /path/to/experiment --convert --lm-eval --lmms-eval"
    echo ""
    echo "  # HF model (no conversion)"
    echo "  $0 meta-llama/Llama-3.2-3B --lm-eval --instruct"
    exit 1
}

# Parse arguments
if [ "$#" -lt 1 ]; then
    echo -e "${RED}ERROR: No experiment path provided${NC}"
    usage
fi

EXPERIMENT_PATH="$1"
shift

# Initialize flags
DO_CONVERT="false"
DO_LM_EVAL="false"
DO_MLLM_EVAL="false"
LM_TASK_GROUPS_FILE=""
MLLM_TASK_GROUPS_FILE=""
INSTRUCT_FLAG=""
BATCH_SIZE=""
TOKENIZER=""
MODEL=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --convert)
            DO_CONVERT="true"
            shift
            ;;
        --lm-eval)
            DO_LM_EVAL="true"
            shift
            ;;
        --lmms-eval)
            DO_MLLM_EVAL="true"
            shift
            ;;
        --lm-task-groups)
            LM_TASK_GROUPS_FILE="$2"
            shift 2
            ;;
        --mllm-task-groups)
            MLLM_TASK_GROUPS_FILE="$2"
            shift 2
            ;;
        --instruct)
            INSTRUCT_FLAG="--instruct"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}ERROR: Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate at least one action is specified
if [ "$DO_CONVERT" = "false" ] && [ "$DO_LM_EVAL" = "false" ] && [ "$DO_MLLM_EVAL" = "false" ]; then
    echo -e "${RED}ERROR: No action specified. Use --convert, --lm-eval, and/or --lmms-eval${NC}"
    usage
fi

# Detect if experiment path is HF identifier or local path
IS_HF_IDENTIFIER="false"
if [[ ! "$EXPERIMENT_PATH" =~ ^[/.] ]] && [[ ! -d "$EXPERIMENT_PATH" ]]; then
    IS_HF_IDENTIFIER="true"
fi

# Validate experiment path
if [ "$IS_HF_IDENTIFIER" = "false" ] && [ ! -d "$EXPERIMENT_PATH" ]; then
    echo -e "${RED}ERROR: Experiment path does not exist: ${EXPERIMENT_PATH}${NC}"
    exit 1
fi

# Validate conversion request for HF identifiers
if [ "$DO_CONVERT" = "true" ] && [ "$IS_HF_IDENTIFIER" = "true" ]; then
    echo -e "${RED}ERROR: Cannot convert HuggingFace model identifier: ${EXPERIMENT_PATH}${NC}"
    echo "HuggingFace models do not require conversion. Remove --convert flag."
    exit 1
fi

# Load task groups
if [ "$DO_LM_EVAL" = "true" ]; then
    if [ -n "$LM_TASK_GROUPS_FILE" ]; then
        if [ ! -f "$LM_TASK_GROUPS_FILE" ]; then
            echo -e "${RED}ERROR: LM task groups file not found: ${LM_TASK_GROUPS_FILE}${NC}"
            exit 1
        fi
        # Validate JSON
        if ! jq empty "$LM_TASK_GROUPS_FILE" 2>/dev/null; then
            echo -e "${RED}ERROR: Invalid JSON in LM task groups file: ${LM_TASK_GROUPS_FILE}${NC}"
            exit 1
        fi
        LM_TASK_GROUPS=$(cat "$LM_TASK_GROUPS_FILE")
    else
        LM_TASK_GROUPS="$DEFAULT_LM_TASKS"
    fi
fi

if [ "$DO_MLLM_EVAL" = "true" ]; then
    if [ -n "$MLLM_TASK_GROUPS_FILE" ]; then
        if [ ! -f "$MLLM_TASK_GROUPS_FILE" ]; then
            echo -e "${RED}ERROR: MLLM task groups file not found: ${MLLM_TASK_GROUPS_FILE}${NC}"
            exit 1
        fi
        # Validate JSON
        if ! jq empty "$MLLM_TASK_GROUPS_FILE" 2>/dev/null; then
            echo -e "${RED}ERROR: Invalid JSON in MLLM task groups file: ${MLLM_TASK_GROUPS_FILE}${NC}"
            exit 1
        fi
        MLLM_TASK_GROUPS=$(cat "$MLLM_TASK_GROUPS_FILE")
    else
        MLLM_TASK_GROUPS="$DEFAULT_MLLM_TASKS"
    fi
fi

# Count total jobs
TOTAL_JOBS=0
if [ "$DO_CONVERT" = "true" ]; then
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
fi
if [ "$DO_LM_EVAL" = "true" ]; then
    NUM_LM_GROUPS=$(echo "$LM_TASK_GROUPS" | jq 'length')
    TOTAL_JOBS=$((TOTAL_JOBS + NUM_LM_GROUPS))
fi
if [ "$DO_MLLM_EVAL" = "true" ]; then
    NUM_MLLM_GROUPS=$(echo "$MLLM_TASK_GROUPS" | jq 'length')
    TOTAL_JOBS=$((TOTAL_JOBS + NUM_MLLM_GROUPS))
fi

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Benchmark Orchestration${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Experiment:        ${EXPERIMENT_PATH}"
echo "Is HF identifier:  ${IS_HF_IDENTIFIER}"
echo "Convert:           ${DO_CONVERT}"
if [ "$DO_LM_EVAL" = "true" ]; then
    echo "LM Eval:           Yes (${NUM_LM_GROUPS} task groups)"
else
    echo "LM Eval:           No"
fi
if [ "$DO_MLLM_EVAL" = "true" ]; then
    echo "MLLM Eval:         Yes (${NUM_MLLM_GROUPS} task groups)"
else
    echo "MLLM Eval:         No"
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# Track job IDs
CONVERT_JOB_ID=""
LM_EVAL_JOB_IDS=()
MLLM_EVAL_JOB_IDS=()
CURRENT_JOB=0

# Launch conversion job
if [ "$DO_CONVERT" = "true" ]; then
    CURRENT_JOB=$((CURRENT_JOB + 1))
    echo -e "${GREEN}[${CURRENT_JOB}/${TOTAL_JOBS}] Launching conversion job...${NC}"

    JOB_OUTPUT=$(sbatch "${SCRIPT_DIR}/submit-convert.slurm" "$EXPERIMENT_PATH")
    CONVERT_JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')

    echo "  Job ID: ${CONVERT_JOB_ID}"
    echo ""
fi

# Launch LM eval jobs
if [ "$DO_LM_EVAL" = "true" ]; then
    NUM_LM_GROUPS=$(echo "$LM_TASK_GROUPS" | jq 'length')

    for i in $(seq 0 $((NUM_LM_GROUPS - 1))); do
        CURRENT_JOB=$((CURRENT_JOB + 1))

        GROUP_NAME=$(echo "$LM_TASK_GROUPS" | jq -r ".[$i].name")
        TASKS=$(echo "$LM_TASK_GROUPS" | jq -r ".[$i].tasks")

        echo -e "${GREEN}[${CURRENT_JOB}/${TOTAL_JOBS}] Launching LM eval job for group '${GROUP_NAME}'${NC}"
        echo "  Tasks: ${TASKS}"

        # Build sbatch command
        SBATCH_CMD="sbatch"

        # Add dependency if conversion job was launched
        if [ -n "$CONVERT_JOB_ID" ]; then
            SBATCH_CMD="$SBATCH_CMD --dependency=afterok:${CONVERT_JOB_ID}"
        fi

        # Build arguments for the eval script
        EVAL_ARGS="$EXPERIMENT_PATH --tasks $TASKS --no-convert"

        if [ -n "$INSTRUCT_FLAG" ]; then
            EVAL_ARGS="$EVAL_ARGS $INSTRUCT_FLAG"
        fi
        if [ -n "$BATCH_SIZE" ]; then
            EVAL_ARGS="$EVAL_ARGS --batch-size $BATCH_SIZE"
        fi
        if [ -n "$TOKENIZER" ]; then
            EVAL_ARGS="$EVAL_ARGS --tokenizer $TOKENIZER"
        fi

        # Submit job
        JOB_OUTPUT=$(eval "$SBATCH_CMD ${SCRIPT_DIR}/submit-convert-and-lm-eval.slurm $EVAL_ARGS")
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
        LM_EVAL_JOB_IDS+=("$JOB_ID")

        if [ -n "$CONVERT_JOB_ID" ]; then
            echo "  Job ID: ${JOB_ID} (depends on: ${CONVERT_JOB_ID})"
        else
            echo "  Job ID: ${JOB_ID} (depends on: none)"
        fi
        echo ""
    done
fi

# Launch MLLM eval jobs
if [ "$DO_MLLM_EVAL" = "true" ]; then
    NUM_MLLM_GROUPS=$(echo "$MLLM_TASK_GROUPS" | jq 'length')

    for i in $(seq 0 $((NUM_MLLM_GROUPS - 1))); do
        CURRENT_JOB=$((CURRENT_JOB + 1))

        GROUP_NAME=$(echo "$MLLM_TASK_GROUPS" | jq -r ".[$i].name")
        TASKS=$(echo "$MLLM_TASK_GROUPS" | jq -r ".[$i].tasks")

        echo -e "${GREEN}[${CURRENT_JOB}/${TOTAL_JOBS}] Launching MLLM eval job for group '${GROUP_NAME}'${NC}"
        echo "  Tasks: ${TASKS}"

        # Build sbatch command
        SBATCH_CMD="sbatch"

        # Add dependency if conversion job was launched
        if [ -n "$CONVERT_JOB_ID" ]; then
            SBATCH_CMD="$SBATCH_CMD --dependency=afterok:${CONVERT_JOB_ID}"
        fi

        # Build arguments for the eval script
        EVAL_ARGS="$EXPERIMENT_PATH --tasks $TASKS --no-convert"

        if [ -n "$BATCH_SIZE" ]; then
            EVAL_ARGS="$EVAL_ARGS --batch-size $BATCH_SIZE"
        fi
        if [ -n "$MODEL" ]; then
            EVAL_ARGS="$EVAL_ARGS --model $MODEL"
        fi

        # Submit job
        JOB_OUTPUT=$(eval "$SBATCH_CMD ${SCRIPT_DIR}/submit-convert-and-mllm-eval.slurm $EVAL_ARGS")
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
        MLLM_EVAL_JOB_IDS+=("$JOB_ID")

        if [ -n "$CONVERT_JOB_ID" ]; then
            echo "  Job ID: ${JOB_ID} (depends on: ${CONVERT_JOB_ID})"
        else
            echo "  Job ID: ${JOB_ID} (depends on: none)"
        fi
        echo ""
    done
fi

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary: Launched ${TOTAL_JOBS} jobs${NC}"
if [ -n "$CONVERT_JOB_ID" ]; then
    echo "  Conversion: ${CONVERT_JOB_ID}"
fi
if [ ${#LM_EVAL_JOB_IDS[@]} -gt 0 ]; then
    echo "  LM Eval: ${LM_EVAL_JOB_IDS[*]}"
fi
if [ ${#MLLM_EVAL_JOB_IDS[@]} -gt 0 ]; then
    echo "  MLLM Eval: ${MLLM_EVAL_JOB_IDS[*]}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: /iopsstor/scratch/cscs/\$USER/PDM/log/"
