#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/log-nemo-Goldfish%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/log-nemo-Goldfish%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=sps_gut
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00

# ---------------------------------------------------
# Environment Setup
# ---------------------------------------------------
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NEMO_LOG_MEMORY_USAGE=1
export WANDB_API_KEY=74bc2e3d0aa09e4d4e8a89659496aa4697714938
export NEMO_TESTING=1

# ---------------------------------------------------
# Training Configuration
# ---------------------------------------------------
num_train_samples=10200350
num_gpus=$((SLURM_JOB_NUM_NODES * 4))
micro_batch_size=3 # maximum micro-batch size that fits into GPU memory
global_batch_size=$((num_gpus * micro_batch_size))
max_steps=$(echo "$num_train_samples / $global_batch_size" | bc)

llama_param_size='1.5B'
goldfish_loss=false
goldfish_h=13
goldfish_k=50

# Set loss type based on goldfish_loss
if [ "$goldfish_loss" = true ]; then
    loss_type="Goldfish"
    run_name="llama_${llama_param_size}_Sparse_Gutenberg_K_${goldfish_k}_H_${goldfish_h}_GBS_${global_batch_size}"
else
    loss_type="Standard"
    run_name="llama_${llama_param_size}_Sparse_Gutenberg_Standard_GBS_${global_batch_size}"
fi

results_dir="/iopsstor/scratch/cscs/xyixuan/experiment/${run_name}"
ckpts_dir="${results_dir}/results/checkpoints"
latest_ckpt=$(ls -d ${ckpts_dir}/*  | sort -t= -k2,2nr | head -n1)
RESUME_OPTION=""
if [ -n "$latest_ckpt" ]; then
    # URL encode the equals signs in the checkpoint path
    encoded_ckpt=$(echo "$latest_ckpt" | sed 's/=/\\=/g')
    RESUME_OPTION="'++exp_manager.resume_from_checkpoint=${encoded_ckpt}'"
    echo "Resuming from checkpoint: $latest_ckpt"
fi

# ---------------------------------------------------
# Output Configuration for Verification
# ---------------------------------------------------
echo "Run Name: $run_name"
echo "Results Directory: $results_dir"
echo "Every N Train Steps: $every_n_train_steps"
echo "Max Steps: $max_steps"

# ---------------------------------------------------
# Execute Training
# ---------------------------------------------------

srun --output /capstor/users/cscs/xyixuan/PDM/log/log-nemo-Goldfish%j.out --error /capstor/users/cscs/xyixuan/PDM/log/log-nemo-Goldfish%j.err --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID --wait 60 --unbuffered bash -c "
  wandb login --relogin 74bc2e3d0aa09e4d4e8a89659496aa4697714938;
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/capstor/users/cscs/xyixuan/PDM/config \
  --config-name=nemo_sparse_gutenberg-hydra.yaml \
  llama_param_size='${llama_param_size}' \
  run.name='${run_name}' \
  run.results_dir='${results_dir}' \
  trainer.num_nodes='${SLURM_JOB_NUM_NODES}' \
  model.micro_batch_size='${micro_batch_size}' \
  model.global_batch_size=${global_batch_size} \
  model.data.goldfish_loss=${goldfish_loss} \
  model.data.goldfish_h=${goldfish_h} \
  model.data.goldfish_k=${goldfish_k} \
  model.gc_interval=100 \
  exp_manager.checkpoint_callback_params.every_n_train_steps=1000 \
  trainer.max_steps=${max_steps} \
  ${RESUME_OPTION} 
"