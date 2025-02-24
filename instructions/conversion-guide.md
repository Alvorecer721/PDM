# Megatron to HuggingFace Checkpoint Conversion Guide

This guide outlines the process of converting Megatron checkpoints to HuggingFace format.

## Step 1: Convert Megatron Checkpoint from `torch_dist` to `torch`

First, modify your pretraining launch script to include checkpoint conversion:

1. Update your script configuration:
```shell
# Remove or comment out these lines
# export WANDB_API_KEY=...
# Remove --async-save from CHECKPOINTING_ARGS

# Add converter arguments
CONVERTER_ARGS=(
  --ckpt-convert-format torch
  --ckpt-convert-save $EXP_DIR
)
```

2. Update the training command to include converter arguments:
```shell
TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${CONVERTER_ARGS[@]} \
    $DATA_ARGS"
```

**Pro Tip:** Keep `CONVERTER_ARGS` commented in your launch script for future reference.

## Step 2: Convert to HuggingFace Format

Run this conversion on a computing node using the NGC PyTorch environment:

```shell
# Set up environment
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Run conversion script
python /capstor/users/cscs/xyixuan/PDM/src/infer/convert_megatron_to_hf.py \
    --experiment-path ${EXPR_PATH}

# EXPR_PATH is the same as in your launch script
```

**Important:** Use the NGC PyTorch environment:
```shell
--environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
```

## Step 3: Evaluation Setup

Install and run the evaluation harness:

```shell
# Setup evaluation directory
EVAL_DIR="/capstor/users/cscs/$USER/lm-evaluation-harness"
if [ ! -d "$EVAL_DIR" ]; then
    echo "Creating lm-evaluation-harness directory..."
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$EVAL_DIR"
fi

# Install lm-eval if needed
if ! pip show lm-eval &> /dev/null; then
    echo "Installing lm-eval..."
    cd "$EVAL_DIR"
    pip install -e .
else
    echo "lm-eval is already installed"
fi

# Run evaluation
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${EXPR_PATH}/HF,tokenizer=meta-llama/Llama-3.1-8B-Instruct \
    --tasks hellaswag,mmlu \
    --batch_size 1 \
    --output_path ${RES_PATH}
```

## Additional Resources

- Combined conversion and evaluation script: [swissai-convert-and-downstream-eval.sh](https://github.com/Alvorecer721/PDM/blob/main/submissions/swissai-convert-and-downstream-eval.sh)
- Computing node access guide: [clariden-remote-debug.md](https://github.com/Alvorecer721/PDM/blob/main/instructions/todi-remote-debug.md)
- Checkpoint conversion helper script: [convert_megatron_to_hf.py](https://github.com/Alvorecer721/PDM/blob/main/src/infer/convert_megatron_to_hf.py)