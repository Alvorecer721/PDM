#!/bin/bash
# SLURM job for standalone logits comparison (no conversion)
#
# Runs compare_logits.py inside a container with GPU access.
# Supports hf:, meg:, and pt: checkpoint specs.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/$USER/PDM/submissions/submit-compare-logits.slurm \
#       <spec1> [spec2 ...] [OPTIONS]
#
# Examples:
#   # Smoke test a single HF model
#   sbatch submit-compare-logits.slurm hf:/path/to/hf_model
#
#   # Compare two HF models
#   sbatch submit-compare-logits.slurm hf:/path/to/model_a hf:/path/to/model_b
#
#   # Compare HF vs Megatron checkpoint
#   sbatch submit-compare-logits.slurm \
#       hf:/path/to/hf_model \
#       meg:/path/to/megatron_ckpt \
#       --megatron-dir /path/to/Megatron-LM \
#       --tokenizer /path/to/tokenizer
#
# Options (passed through to compare_logits.py):
#   --prompt TEXT              Prompt for extraction (default: "Sanity check prompt.")
#   --dtype bf16|fp16|fp32     Model dtype (default: bf16)
#   --atol FLOAT               Absolute tolerance (default: 1e-2)
#   --rtol FLOAT               Relative tolerance (default: 1e-2)
#   --trust-remote-code        For HF models with custom code
#   --tokenizer PATH           Explicit tokenizer for HF models
#   --megatron-dir PATH        Path to Megatron-LM repo
#   --old-megatron          Use old pretrain_gpt model_provider for meg: specs
#
# SLURM-only options (consumed by this script, not passed to Python):
#   --container-env PATH       Container .toml to use (default: see CONTAINER_ENV below)

#SBATCH --account=infra01
#SBATCH --gres=gpu:4
#SBATCH --job-name=compare-logits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/PDM/log/slurm/compare_logits_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/PDM/log/slurm/compare_logits_%j.err

set -e
set -x
ulimit -c 0

# Default container environment
CONTAINER_ENV=/users/rkreft/.edf/new_ngc.toml
SRUN_CUSTOM_FLAGS="--mpi=pmix --network=disable_rdzv_get"

PDM_DIR=/iopsstor/scratch/cscs/$USER/PDM

if [ "$#" -lt 1 ]; then
    echo "Error: No checkpoint specs provided!"
    echo "Usage: sbatch $0 <spec1> [spec2 ...] [OPTIONS]"
    echo ""
    echo "Specs:  hf:/path/to/hf_model  meg:/path/to/megatron_ckpt  pt:/path/to/logits.pt"
    echo ""
    echo "See script header for full options list."
    exit 1
fi

# Parse out --container-env (consumed here, not passed to Python)
COMPARE_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --container-env)
            CONTAINER_ENV="$2"
            shift 2
            ;;
        *)
            COMPARE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Setup logging
LOG_DIR="/iopsstor/scratch/cscs/$USER/PDM/log/compare_logits"
mkdir -p "$LOG_DIR" || { echo "Failed to create log directory"; exit 1; }
exec > "$LOG_DIR/compare_logits_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).out" 2>&1

echo "=========================================================="
echo "Logits Comparison - Job ${SLURM_JOB_ID}"
echo "Container:  ${CONTAINER_ENV}"
echo "Arguments:  ${COMPARE_ARGS[*]}"
echo "=========================================================="

SCRIPT_PATH="${PDM_DIR}/src/convert/compare_logits.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: compare_logits.py not found at $SCRIPT_PATH"
    exit 1
fi

# Ensure transformers >= 4.56.0 inside the container
ensure_transformers() {
    local required="4.56.0"
    local current
    current=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "")
    if [ -z "$current" ]; then
        echo "transformers not found, installing transformers>=${required}..."
        pip install "transformers>=4.56,<5.0.0"
    elif python -c "from packaging import version; exit(0 if version.parse('${current}') >= version.parse('${required}') else 1)"; then
        echo "transformers version ${current} >= ${required} OK"
    else
        echo "transformers version ${current} < ${required}, upgrading..."
        pip install -U "transformers>=4.56,<5.0.0"
    fi
}
export -f ensure_transformers

srun $SRUN_CUSTOM_FLAGS --environment="$CONTAINER_ENV" \
    bash -c 'ensure_transformers && python "$@"' _ "$SCRIPT_PATH" "${COMPARE_ARGS[@]}"

echo "Comparison End"
