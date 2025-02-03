#!/bin/bash

# Get the master node hostname
# MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# export MASTER_PORT
# export MASTER_ADDR

export PYTHONPATH="/capstor/users/cscs/xyixuan/PDM:${PYTHONPATH}"

EXPR_PATH="/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60"
EXPR_NAME=$(basename ${EXPR_PATH})
RES_PATH="/capstor/users/cscs/xyixuan/PDM/results/lm_eval/${EXPR_NAME}"
mkdir -p ${RES_PATH}

python /capstor/users/cscs/xyixuan/PDM/src/lm_eval/prepare_model.py \
   --config /capstor/users/cscs/xyixuan/PDM/config/llama3_1.5B_config.json \
   --expr ${EXPR_PATH}

# Check if the conversion was successful
if [ $? -ne 0 ]; then
   echo "Model conversion failed"
   exit 1
fi

# Then run your evaluation command
accelerate launch -m lm_eval --model hf \
   --model_args pretrained=${EXPR_PATH}/results/HF,tokenizer=meta-llama/Llama-3.1-8B-Instruct \
   --tasks hellaswag,mmlu \
   --batch_size 64 \
   --output_path ${RES_PATH}
