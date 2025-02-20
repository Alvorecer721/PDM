#!/bin/bash

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

EXPR_PATH="/iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/Goldfish/llama3-1b-15n-8192sl-60gbsz-standard-no-bos"
EXPR_NAME=$(basename ${EXPR_PATH})
RES_PATH="/capstor/users/cscs/xyixuan/PDM/results/lm_eval/${EXPR_NAME}"
mkdir -p ${RES_PATH}

python /capstor/users/cscs/xyixuan/PDM/src/infer/convert_megatron_to_hf.py \
   --experiment-path ${EXPR_PATH}

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion failed"
   exit 1
fi

# Check if lm-eval is already installed
if ! pip show lm-eval &> /dev/null; then
    echo "Installing lm-eval..."
    cd /capstor/users/cscs/xyixuan/lm-evaluation-harness
    pip install -e .
else
    echo "lm-eval is already installed, skipping installation"
fi

# Then run your evaluation command
accelerate launch -m lm_eval --model hf \
   --model_args pretrained=${EXPR_PATH}/HF,tokenizer=meta-llama/Llama-3.1-8B-Instruct \
   --tasks hellaswag,mmlu \
   --batch_size 1 \
   --output_path ${RES_PATH}

# ATTENTION: 
# this script should be run inside computing node with:
# --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml 