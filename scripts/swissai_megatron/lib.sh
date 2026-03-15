#!/bin/bash
# Shared utility functions for Megatron conversion and evaluation scripts.
# Source this file: source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Default prompt for logits comparison (shared across convert.sh and convert_apertus.sh)
DEFAULT_LOGITS_PROMPT="Sanity check prompt."

# Detect tensor-parallel size from a Megatron checkpoint directory.
# Sets TP_SIZE variable in the caller's scope.
# Args: $1 = checkpoint_path
# Returns 0 on success (TP_SIZE set), always sets TP_SIZE (defaults to 1).
detect_tp_size() {
    local checkpoint_path="$1"
    TP_SIZE=1

    local latest_iter
    latest_iter=$(cat "${checkpoint_path}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
    local iter_dir=""
    if [ -n "$latest_iter" ]; then
        iter_dir="${checkpoint_path}/iter_$(printf '%07d' "$latest_iter")"
    fi

    if [ -z "$iter_dir" ] || [ ! -d "$iter_dir" ]; then
        # Try to find any iter directory
        iter_dir=$(find "${checkpoint_path}" -maxdepth 1 -type d -name "iter_*" 2>/dev/null | head -1)
    fi

    if [ -z "$iter_dir" ] || [ ! -d "$iter_dir" ]; then
        echo "WARNING: Could not find checkpoint iteration directory in ${checkpoint_path}, defaulting to TP=1"
        return 0
    fi

    local common_pt="${iter_dir}/common.pt"
    if [ -f "$common_pt" ]; then
        TP_SIZE=$(python -c "
import torch
d = torch.load('${common_pt}', weights_only=False, map_location='cpu')
args = d.get('args')
if args:
    print(getattr(args, 'tensor_model_parallel_size', 1))
else:
    print(1)
" 2>/dev/null || echo "1")
        echo "Detected tensor_model_parallel_size=${TP_SIZE} from checkpoint"
    else
        echo "WARNING: common.pt not found at ${common_pt}, defaulting to TP=1"
    fi
}

# Run logits comparison between an HF model and optionally a reference.
# Args:
#   $1 = hf_dir            Path to HF model directory
#   $2 = checkpoint_path   Path to source Megatron checkpoint
#   $3 = default_prompt    Prompt string for logits extraction
#   $4 = megatron_dir      Path to Megatron-LM repo
#   $5 = pdm_dir           Path to PDM repo
#   $6 = old_megatron      "true" to pass --old-megatron, anything else to skip
#   $7 = logits_test_ref   Optional reference spec (e.g. hf:/path or meg:/path)
#   $8 = tokenizer_path    Optional explicit tokenizer path for HF models
# Returns: exit code from compare_logits.py (0=pass, non-zero=fail)
run_logits_comparison() {
    local hf_dir="$1"
    local checkpoint_path="$2"
    local default_prompt="$3"
    local megatron_dir="$4"
    local pdm_dir="$5"
    local old_megatron="$6"
    local logits_test_ref="$7"
    local tokenizer_path="$8"

    echo ""
    echo "========================================"
    echo "Running logits comparison..."
    echo "========================================"

    local -a logits_cmd=(
        python "${pdm_dir}/src/convert/compare_logits.py"
        "hf:${hf_dir}"
    )

    if [ -n "$logits_test_ref" ]; then
        logits_cmd+=("$logits_test_ref")
    else
        logits_cmd+=("meg:${checkpoint_path}")
    fi

    logits_cmd+=(--prompt "$default_prompt" --dtype bf16 --megatron-dir "$megatron_dir")

    if [ -n "$tokenizer_path" ]; then
        logits_cmd+=(--tokenizer "$tokenizer_path")
    fi

    if [ "$old_megatron" = "true" ]; then
        logits_cmd+=(--old-megatron)
    fi

    "${logits_cmd[@]}" && local logits_exit=0 || local logits_exit=$?
    if [ $logits_exit -ne 0 ]; then
        echo "WARNING: Logits comparison failed (exit code $logits_exit)"
        echo "The HF conversion itself succeeded, but logits do not match."
    else
        echo "Logits comparison passed."
    fi
    return $logits_exit
}
