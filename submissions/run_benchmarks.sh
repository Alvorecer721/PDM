#!/bin/bash

# Orchestrator script for launching multiple benchmarking jobs with task groups
# Supports standalone conversion, LM evaluation, and MLLM evaluation with SLURM dependencies
# Uses submit-convert.slurm, submit-convert-and-lm-eval.slurm and submit-convert-and-mllm-eval.slurm 
#
# Path arguments:
#   experiment_path  - Output directory for converted models (torch/, HF/) and results
#   checkpoint_path  - Source torch-dist checkpoint (default: ${experiment_path}/checkpoints/3B)

set -e  # Exit on error
ulimit -c 0 # no core dumps

# Color codes for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default task groups
DEFAULT_LM_TASKS='[{"name":"default","tasks":"hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa"}]'
DEFAULT_MLLM_TASKS='[{"name":"default","tasks":"pope,gqa,vqav2_val,mmmu_val_group_img,mme,ai2d,ocrbench_v2,chartqa,docvqa,infovqa"}]'

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
    echo "Model Options:"
    echo "  --model-type TYPE              Model architecture type (default: llama3)"
    echo "                                 Options: llama3, apertus"
    echo "  --checkpoint-path PATH         Direct path to torch distributed checkpoint"
    echo "                                 (default: \${experiment_path}/checkpoints/3B) -> MUST define for apertus model!!"
    echo "  --debug                         Adds debug prints to the first 5 processed samples of each spawned eval task for mllm eval"
    echo ""
    echo "Evaluation Options:"
    echo "  --instruct                     Use instruct mode for LM eval (sets chat template)"
    echo "  --lm-batch-size SIZE           Batch size for LM evaluation (default: 6)"
    echo "  --mllm-batch-size SIZE         Batch size for MLLM evaluation (default: 3)"
    echo "  --tokenizer TOKENIZER          Tokenizer to use"
    echo "  --model MODEL                  MLLM model type (MLLM eval only)"
    echo ""
    echo "Wandb Options:"
    echo "  --no-offline-datasets           Disable offline mode for HF datasets (default: offline mode enabled)"
    echo "  --no-wandb                     Disable wandb logging (default: enabled)"
    echo "  --wandb-project PROJECT        Wandb project name (default: lm-eval for LM, lmms-eval for MLLM)"
    echo ""
    echo "Environment Options:"
    echo "  --environment ENV              SLURM environment to use (default: nemo_container)"
    echo "                                 Can be a container name or path to environment file"
    echo ""
    echo "Dependency Options:"
    echo "  --dependency JOB_ID            SLURM job ID to wait for before starting conversion"
    echo "                                 Uses afterany relation (runs regardless of job status)"
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
    echo "  # Full pipeline with custom environment"
    echo "  $0 /path/to/experiment --convert --lm-eval --lmms-eval --environment /path/to/environment.toml"
    echo ""
    echo "  # Complete example with llama3 model and task groups"
    echo "  $0 /path/to/experiment"
    echo "    --convert"
    echo "    --lm-eval --lm-task-groups /iopsstor/scratch/cscs/\$USER/PDM/config/lm_eval_task_groups.json"
    echo "    --lmms-eval --mllm-task-groups /iopsstor/scratch/cscs/\$USER/PDM/config/mllm_eval_task_groups.json"
    echo "    --model-type llama3"
    echo "    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer"
    echo "    --environment nemo_container"
    echo ""
    echo "  # Complete example with apertus model (single tasks)"
    echo "  $0 /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model2"
    echo "    --convert"
    echo "    --checkpoint-path /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model2/iter_0000001"
    echo "    --lm-eval --lm-task-groups /iopsstor/scratch/cscs/\$USER/PDM/config/lm_eval_task_groups.json"
    echo "    --lmms-eval --mllm-task-groups /iopsstor/scratch/cscs/\$USER/PDM/config/mllm_eval_task_groups.json"
    echo "    --model-type apertus"
    echo "    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer"
    echo "    --model apertus_emu3p5_simple"
    echo ""
    echo "  # HF model (no conversion)"
    echo "  $0 meta-llama/Llama-3.2-3B --lm-eval --instruct"
    echo ""
    echo "  # Testing/debugging with test configs and debug mode (smaller task sets for quick validation)"
    echo "  $0 /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model2"
    echo "    --convert"
    echo "    --checkpoint-path /capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model2/iter_0000001"
    echo "    --lm-eval --lm-task-groups /iopsstor/scratch/cscs/\$USER/PDM/config/lm_eval_apertus_test.json"
    echo "    --lmms-eval --mllm-task-groups /iopsstor/scratch/cscs/\$USER/PDM/config/mllm_eval_apertus_test.json"
    echo "    --model-type apertus"
    echo "    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/apertus_emu3.5_tokenizer"
    echo "    --model apertus_emu3p5_simple"
    echo "    --debug"
    echo ""
    echo "Task Group Files:"
    echo "  Predefined task group configurations are available in:"
    echo "  /iopsstor/scratch/cscs/\$USER/PDM/config/"
    echo "  - lm_task_groups.json: Default LM evaluation task groups"
    echo "  - mllm_task_groups.json: Default MLLM evaluation task groups"
    echo "  - lm_eval_apertus_test.json: Test config with minimal tasks (winogrande)"
    echo "  - mllm_eval_apertus_test.json: Test config with minimal tasks (mmmu_val_group_img)"
    echo "  You can use these files directly or create custom ones."
    exit 1
}

# Validation function - validates all inputs and loads task group configurations
# Sets: IS_HF_IDENTIFIER, LM_TASK_GROUPS, MLLM_TASK_GROUPS
validate() {
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

    # Validate experiment path exists (for local paths)
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

    # Validate model type
    if [ "$MODEL_TYPE" != "llama3" ] && [ "$MODEL_TYPE" != "apertus" ]; then
        echo -e "${RED}ERROR: Invalid model type: ${MODEL_TYPE}${NC}"
        echo "Valid options: llama3, apertus"
        exit 1
    fi

    # Load and validate LM task groups
    if [ "$DO_LM_EVAL" = "true" ]; then
        if [ -n "$LM_TASK_GROUPS_FILE" ]; then
            if [ ! -f "$LM_TASK_GROUPS_FILE" ]; then
                echo -e "${RED}ERROR: LM task groups file not found: ${LM_TASK_GROUPS_FILE}${NC}"
                exit 1
            fi
            if ! jq empty "$LM_TASK_GROUPS_FILE" 2>/dev/null; then
                echo -e "${RED}ERROR: Invalid JSON in LM task groups file: ${LM_TASK_GROUPS_FILE}${NC}"
                exit 1
            fi
            LM_TASK_GROUPS=$(cat "$LM_TASK_GROUPS_FILE")
        else
            LM_TASK_GROUPS="$DEFAULT_LM_TASKS"
        fi
    fi

    # Load and validate MLLM task groups
    if [ "$DO_MLLM_EVAL" = "true" ]; then
        if [ -n "$MLLM_TASK_GROUPS_FILE" ]; then
            if [ ! -f "$MLLM_TASK_GROUPS_FILE" ]; then
                echo -e "${RED}ERROR: MLLM task groups file not found: ${MLLM_TASK_GROUPS_FILE}${NC}"
                exit 1
            fi
            if ! jq empty "$MLLM_TASK_GROUPS_FILE" 2>/dev/null; then
                echo -e "${RED}ERROR: Invalid JSON in MLLM task groups file: ${MLLM_TASK_GROUPS_FILE}${NC}"
                exit 1
            fi
            MLLM_TASK_GROUPS=$(cat "$MLLM_TASK_GROUPS_FILE")
        else
            MLLM_TASK_GROUPS="$DEFAULT_MLLM_TASKS"
        fi
    fi
}

# Check for help flags before argument validation
for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        usage
    fi
done

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
MODEL_TYPE="llama3"
CHECKPOINT_PATH=""
INSTRUCT_FLAG=""
LM_BATCH_SIZE="6"
MLLM_BATCH_SIZE="3"
TOKENIZER=""
MODEL=""
DEBUG=""
OFFLINE_DATASETS="true"
WANDB_ENABLED="true"
WANDB_PROJECT=""
ENVIRONMENT="/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml"
DEPENDENCY_JOB=""
#ENVIRONMENT="/capstor/store/cscs/swissai/infra01/containers/nemo.toml"

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
        --debug)
            DEBUG="true"
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
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --instruct)
            INSTRUCT_FLAG="--apply-chat-template" # only set apply chat template, as --instruct would also set tokenizer and instruct mode
            shift
            ;;
        --lm-batch-size)
            LM_BATCH_SIZE="$2"
            shift 2
            ;;
        --mllm-batch-size)
            MLLM_BATCH_SIZE="$2"
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
        --no-offline-datasets)
            OFFLINE_DATASETS="false"
            shift
            ;;
        --no-wandb)
            WANDB_ENABLED="false"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --dependency)
            DEPENDENCY_JOB="$2"
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

# Validate all inputs and load task groups
validate

# Generate unique log subdirectory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
# Use last two path components joined with __ for a meaningful log dir name
CLEAN_EXP_PATH="${EXPERIMENT_PATH%/}"
EXPERIMENT_BASENAME="$(basename "$(dirname "$CLEAN_EXP_PATH")")__$(basename "$CLEAN_EXP_PATH")"
LOG_SUBDIR="log/run_${TIMESTAMP}_${EXPERIMENT_BASENAME}"
LOG_DIR="${SCRIPT_DIR}/../${LOG_SUBDIR}"
mkdir -p "$LOG_DIR"

# Ensure SLURM log directory exists
mkdir -p "${SCRIPT_DIR}/../log/slurm"

echo -e "${BLUE}Log directory: ${LOG_SUBDIR}${NC}"
echo ""

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
echo "Environment:       ${ENVIRONMENT}"
if [ -n "$DEPENDENCY_JOB" ]; then
    echo "Dependency:        ${DEPENDENCY_JOB} (afterany)"
fi
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

    # Build conversion arguments
    CONVERT_ARGS="$EXPERIMENT_PATH --model-type $MODEL_TYPE --log-subdir $LOG_SUBDIR"
    if [ -n "$CHECKPOINT_PATH" ]; then
        CONVERT_ARGS="$CONVERT_ARGS --checkpoint-path $CHECKPOINT_PATH"
    fi
    if [ -n "$TOKENIZER" ]; then
        CONVERT_ARGS="$CONVERT_ARGS --tokenizer $TOKENIZER"
    fi

    # Build sbatch command
    SBATCH_CMD="sbatch"

    # Add dependency if specified
    if [ -n "$DEPENDENCY_JOB" ]; then
        SBATCH_CMD="$SBATCH_CMD --dependency=afterany:${DEPENDENCY_JOB}"
    fi

    # Add environment if specified
    if [ -n "$ENVIRONMENT" ]; then
        SBATCH_CMD="$SBATCH_CMD --environment=$ENVIRONMENT"
    fi

    # Submit job
    JOB_OUTPUT=$(eval "$SBATCH_CMD ${SCRIPT_DIR}/submit-convert.slurm $CONVERT_ARGS")
    CONVERT_JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')

    if [ -n "$DEPENDENCY_JOB" ]; then
        echo "  Job ID: ${CONVERT_JOB_ID} (depends on: ${DEPENDENCY_JOB})"
    else
        echo "  Job ID: ${CONVERT_JOB_ID}"
    fi
    echo "  Environment: ${ENVIRONMENT}"
    echo ""
fi

 # if this script was converting based on the experiment path, the actual checkpoint is under $EXPERIMENT_PATH/HF
EVAL_CP_PATH="$EXPERIMENT_PATH"
if [ -n "$CONVERT_JOB_ID" ]; then
    EVAL_CP_PATH="$EVAL_CP_PATH/HF"
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
        echo "  Environment: ${ENVIRONMENT}"

        # Build sbatch command
        SBATCH_CMD="sbatch"

        # Add dependency if conversion job was launched
        if [ -n "$CONVERT_JOB_ID" ]; then
            SBATCH_CMD="$SBATCH_CMD --dependency=afterok:${CONVERT_JOB_ID}"
        fi

        # Add environment if specified
        if [ -n "$ENVIRONMENT" ]; then
            SBATCH_CMD="$SBATCH_CMD --environment=$ENVIRONMENT"
        fi

        # Build arguments for the eval script
        EVAL_ARGS="$EVAL_CP_PATH --tasks $TASKS --model-type $MODEL_TYPE --log-subdir $LOG_SUBDIR --group-name $GROUP_NAME"
        
        # never convert as this is done by this script potentially if needed
        EVAL_ARGS="$EVAL_ARGS --no-convert"

        if [ -n "$INSTRUCT_FLAG" ]; then
            EVAL_ARGS="$EVAL_ARGS $INSTRUCT_FLAG"
        fi
        EVAL_ARGS="$EVAL_ARGS --batch-size $LM_BATCH_SIZE"
        if [ -n "$TOKENIZER" ]; then
            EVAL_ARGS="$EVAL_ARGS --tokenizer $TOKENIZER"
        fi
        if [ "$WANDB_ENABLED" = "false" ]; then
            EVAL_ARGS="$EVAL_ARGS --no-wandb"
        fi
        if [ -n "$WANDB_PROJECT" ]; then
            EVAL_ARGS="$EVAL_ARGS --wandb-project $WANDB_PROJECT"
        fi
        if [ "$OFFLINE_DATASETS" = "false" ]; then
            EVAL_ARGS="$EVAL_ARGS --no-offline-datasets"
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
        echo "  Environment: ${ENVIRONMENT}"

        # Build sbatch command
        SBATCH_CMD="sbatch"

        # Add dependency if conversion job was launched
        if [ -n "$CONVERT_JOB_ID" ]; then
            SBATCH_CMD="$SBATCH_CMD --dependency=afterok:${CONVERT_JOB_ID}"
        fi

        # Add environment if specified
        if [ -n "$ENVIRONMENT" ]; then
            SBATCH_CMD="$SBATCH_CMD --environment=$ENVIRONMENT"
        fi

        # Build arguments for the eval script
        EVAL_ARGS="$EVAL_CP_PATH --no-convert --tasks $TASKS --model-type $MODEL_TYPE --log-subdir $LOG_SUBDIR --group-name $GROUP_NAME"

        EVAL_ARGS="$EVAL_ARGS --batch-size $MLLM_BATCH_SIZE"
        if [ -n "$TOKENIZER" ]; then
            EVAL_ARGS="$EVAL_ARGS --tokenizer $TOKENIZER"
        fi
        if [ -n "$MODEL" ]; then
            EVAL_ARGS="$EVAL_ARGS --model $MODEL"
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
        if [ "$OFFLINE_DATASETS" = "false" ]; then
            EVAL_ARGS="$EVAL_ARGS --no-offline-datasets"
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

# Launch summary job
echo -e "${GREEN}Launching summary job...${NC}"

# Collect all job IDs for dependency
ALL_JOB_IDS=()
if [ -n "$CONVERT_JOB_ID" ]; then
    ALL_JOB_IDS+=("$CONVERT_JOB_ID")
fi
ALL_JOB_IDS+=("${LM_EVAL_JOB_IDS[@]}")
ALL_JOB_IDS+=("${MLLM_EVAL_JOB_IDS[@]}")

# Build dependency string
DEPENDENCY_STRING=""
if [ ${#ALL_JOB_IDS[@]} -gt 0 ]; then
    DEPENDENCY_STRING="--dependency=afterany:$(IFS=:; echo "${ALL_JOB_IDS[*]}")"
fi

# Build summary job arguments
SUMMARY_ARGS="$LOG_SUBDIR"
if [ -n "$CONVERT_JOB_ID" ]; then
    SUMMARY_ARGS="$SUMMARY_ARGS --convert $CONVERT_JOB_ID"
fi
if [ ${#LM_EVAL_JOB_IDS[@]} -gt 0 ]; then
    SUMMARY_ARGS="$SUMMARY_ARGS --lm-eval ${LM_EVAL_JOB_IDS[*]}"
fi
if [ ${#MLLM_EVAL_JOB_IDS[@]} -gt 0 ]; then
    SUMMARY_ARGS="$SUMMARY_ARGS --mllm-eval ${MLLM_EVAL_JOB_IDS[*]}"
fi

# Submit summary job
SUMMARY_CMD="sbatch $DEPENDENCY_STRING"
SUMMARY_OUTPUT=$(eval "$SUMMARY_CMD ${SCRIPT_DIR}/submit-job-summary.slurm $SUMMARY_ARGS")
SUMMARY_JOB_ID=$(echo "$SUMMARY_OUTPUT" | grep -oP '\d+$')

echo "  Summary Job ID: ${SUMMARY_JOB_ID}"
echo ""

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary: Launched ${TOTAL_JOBS} jobs + summary${NC}"
if [ -n "$CONVERT_JOB_ID" ]; then
    echo "  Conversion: ${CONVERT_JOB_ID}"
fi
if [ ${#LM_EVAL_JOB_IDS[@]} -gt 0 ]; then
    echo "  LM Eval: ${LM_EVAL_JOB_IDS[*]}"
fi
if [ ${#MLLM_EVAL_JOB_IDS[@]} -gt 0 ]; then
    echo "  MLLM Eval: ${MLLM_EVAL_JOB_IDS[*]}"
fi
echo "  Summary: ${SUMMARY_JOB_ID}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ${LOG_DIR}"
echo "Summary will be available in: ${LOG_DIR}/summary.txt"
