#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "ERROR: Please provide experiment path"
    echo "Usage: $0 /path/to/experiment/directory"
    exit 1
fi

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
EVAL_DIR=/iopsstor/scratch/cscs/$USER/lm-evaluation-harness

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

EXPR_PATH="$1"
EXPR_NAME=$(basename ${EXPR_PATH})
RES_PATH="/capstor/users/cscs/xyixuan/PDM/results/lm_eval/${EXPR_NAME}"

# Clear existing results directory if it exists
if [ -d "${RES_PATH}" ]; then
    echo "Clearing existing results directory: ${RES_PATH}"
    rm -rf "${RES_PATH}"/*
fi


if [ ! -d "$EVAL_DIR" ]; then
    echo "Creating lm-evaluation-harness directory..."
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$EVAL_DIR"
else
    echo "lm-evaluation-harness directory exists, updating repository..."
    cd "$EVAL_DIR"
    git pull
fi

# Create results directory
mkdir -p ${RES_PATH}

python /capstor/users/cscs/xyixuan/PDM/src/convert/convert_megatron_to_hf.py \
   --experiment-path ${EXPR_PATH}

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion failed"
   exit 1
fi

# Install/update lm-eval
echo "📦 Setting up lm-eval package..."
cd "$EVAL_DIR"
pip install -e .

# Then run your evaluation command
accelerate launch -m lm_eval --model hf \
   --model_args pretrained=${EXPR_PATH}/HF,tokenizer=meta-llama/Llama-3.1-8B-Instruct \
   --tasks hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa \
   --batch_size 4 \
   --output_path ${RES_PATH}

# ATTENTION: 
# this script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml