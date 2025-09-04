"""
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun src/convert/convert_torch_dist_to_torch.py \
    --bf16 \
    --use-precision-aware-optimizer \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --wgrad-deferral-limit 50 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --main-grads-dtype bf16 \
    --attention-dropout 0.0 \
    --use-rope-scaling \
    --swiglu \
    --group-query-attention \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model nvidia/OpenMath2-Llama3.1-8B \
    --make-vocab-size-divisible-by 128 \
    --num-layers 16 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --max-position-embeddings 8192 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --rope-scaling-factor 32 \
    --normalization RMSNorm \
    --seq-length 8192 \
    --load /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz/checkpoints \
    --ckpt-convert-save /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun src/convert/convert_torch_dist_to_torch.py \
    --bf16 \
    --load /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/Offset-Effect/llama3-8b-15n-8192sl-60gbsz-standard/checkpoints \
    --ckpt-convert-save /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/Offset-Effect/llama3-8b-15n-8192sl-60gbsz-standard
    
"""
from megatron.core.enums import ModelType
from megatron.training.training import setup_model_and_optimizer
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
from pretrain_gpt import model_provider


def main():
    
    # Apply ALL model configuration parameters directly 
    args_defaults = {
        "transformer_impl": "transformer_engine",
        "use_checkpoint_args": True,
        "ckpt_format": "torch_dist",
        "ckpt_convert_format": "torch",
        "no_load_rng": True,
        "no_load_optim": True,
        "no_save_optim": True,
        "--untie-embeddings-and-output-weights": False,
        
        # Fake args for initialization
        "micro_batch_size": 1,
        "train_iters": 1,
        "lr": 0.0,
    }

    initialize_megatron(
        args_defaults=args_defaults,
    )
    args = get_args()
    assert args.load is not None, "You must specify --load"
    assert args.ckpt_convert_save is not None, "You must specify --ckpt-convert-save"
    setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

if __name__ == "__main__":
    main()