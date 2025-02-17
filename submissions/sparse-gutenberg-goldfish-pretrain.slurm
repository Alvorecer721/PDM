#!/bin/bash

# Parameters
#SBATCH --account=a-a06
#SBATCH --cpus-per-task=72
#SBATCH --environment=nemo
#SBATCH --error=/capstor/users/cscs/xyixuan/PDM/log/pretrain/log-nemo-Goldfish%j.err
#SBATCH --output=/capstor/users/cscs/xyixuan/PDM/log/pretrain/log-nemo-Goldfish%j.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=sps_gut
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00

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
num_epochs=1

# num_train_samples=1984000 # Only Gutenberg repetition
# num_train_samples=2023680 # FineWeb, Gutenberg 0.02:1
# num_train_samples=2380800 # FineWeb, Gutenberg 0.2:1
# num_train_samples=3968000 # FineWeb, Gutenberg 1:1
# num_train_samples=5952000 # FineWeb, Gutenberg 2:1
# num_train_samples=7936000 # FineWeb, Gutenberg 3:1
# num_train_samples=9920000 # FineWeb, Gutenberg 4:1
# num_train_samples=11971350 # Fineweb + Gutenberg 2048
num_train_samples=10200350 # Fineweb + Gutenberg 128

num_gpus=$((SLURM_JOB_NUM_NODES * 4))
micro_batch_size=1 # maximum micro-batch size that fits into GPU memory: 3
global_batch_size=$((num_gpus * micro_batch_size))
max_steps=$(echo "$num_train_samples / $global_batch_size" | bc)

llama_param_size='3.6B'
goldfish_loss=false
goldfish_h=50
goldfish_k=50

case "$llama_param_size" in
    "1.5B")
        num_layers=16
        hidden_size=2048
        ffn_hidden_size=8192
        num_attention_heads=32
        context_parallel_size=1
        ;;
    "3.6B")
        num_layers=28
        hidden_size=3072
        ffn_hidden_size=8192
        num_attention_heads=24
        context_parallel_size=1
        ;;
    "8B")
        num_layers=32
        hidden_size=4096
        ffn_hidden_size=14336
        num_attention_heads=32
        context_parallel_size=2
        ;;
    *)
        echo "Invalid llama_param_size: $llama_param_size"
        exit 1
        ;;
esac


# Set loss type based on goldfish_loss
if [ "$goldfish_loss" = true ]; then
    loss_type="Goldfish"
    RUN_NAME="llama_${llama_param_size}_Sparse_Gutenberg_K_${goldfish_k}_H_${goldfish_h}_GBS_${global_batch_size}_SEQ_${num_train_samples}"
else
    loss_type="Standard"
    RUN_NAME="llama_${llama_param_size}_Sparse_Gutenberg_Standard_GBS_${global_batch_size}_SEQ_${num_train_samples}"
fi

results_dir="/iopsstor/scratch/cscs/xyixuan/experiment/${RUN_NAME}"
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
echo "Run Name: $RUN_NAME"
echo "Results Directory: $results_dir"
echo "Every N Train Steps: $every_n_train_steps"
echo "Max Steps: $max_steps"

# ---------------------------------------------------
# Execute Training
# ---------------------------------------------------

srun --output /capstor/users/cscs/xyixuan/PDM/log/log-nemo-Goldfish%j.out --error /capstor/users/cscs/xyixuan/PDM/log/log-nemo-Goldfish%j.err --cpus-per-task $SLURM_CPUS_PER_TASK --jobid $SLURM_JOB_ID --wait 60 --unbuffered bash -c "
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/capstor/users/cscs/xyixuan/PDM/config \
  --config-name=nemo_sparse_gutenberg-hydra.yaml \
  llama_param_size='${llama_param_size}' \
  run.name='${RUN_NAME}' \
  run.results_dir='${results_dir}' \
  trainer.num_nodes='${SLURM_JOB_NUM_NODES}' \
  model.micro_batch_size='${micro_batch_size}' \
  model.global_batch_size=${global_batch_size} \
  model.num_layers=${num_layers} \
  model.hidden_size=${hidden_size} \
  model.ffn_hidden_size=${ffn_hidden_size} \
  model.num_attention_heads=${num_attention_heads} \
  model.context_parallel_size=${context_parallel_size} \
  model.data.goldfish_loss=${goldfish_loss} \
  model.data.goldfish_h=${goldfish_h} \
  model.data.goldfish_k=${goldfish_k} \
  model.gc_interval=100 \
  exp_manager.checkpoint_callback_params.every_n_train_steps=15455 \
  trainer.max_steps=${max_steps} \
  ++exp_manager.wandb_logger_kwargs.name=$RUN_NAME-$SLURM_JOB_ID \
  ${RESUME_OPTION} 
"