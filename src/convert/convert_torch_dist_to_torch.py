"""
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun src/convert/convert_torch_dist_to_torch.py \
    --bf16 \
    --use-precision-aware-optimizer \
    --use-distributed-optimizer \
    --main-grads-dtype bf16 \
    --attention-dropout 0.0 \
    --seq-length 8192 \
    --tokenizer-type HuggingFaceTokenizer \
	--tokenizer-model nvidia/OpenMath2-Llama3.1-8B \
    --tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--context-parallel-size 1 \
    --llama 1B \
    --load /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz \
    --ckpt-convert-save /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/DenseGutenberg/llama3-1b-standard-80gbsz
"""
import sys
import argparse
from utils import is_rank_0
from megatron.core.enums import ModelType
from megatron.training.training import setup_model_and_optimizer
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
from pretrain_gpt import model_provider

# Predefined model configurations
MODEL_CONFIGS = {
    "1B": {
        "num_layers": 16,
        "hidden_size": 2048,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "rotary_base": 500000,
        "rope_scaling_factor": 32,
        "norm_epsilon": 1e-5,
        "normalization": "RMSNorm",
        "padded_vocab_size": 128128,
        "untie_embeddings_and_output_weights": False,
        "max_position_embeddings": 8192,
    },
    "3B": {
        "num_layers": 28,
        "hidden_size": 3072,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 24,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "rotary_base": 500000,
        "rope_scaling_factor": 32,
        "norm_epsilon": 1e-5,
        "normalization": "RMSNorm",
        "padded_vocab_size": 128128,
        "untie_embeddings_and_output_weights": False,
        "max_position_embeddings": 8192,
    },
    "8B": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "position_embedding_type": "rope",
        "rotary_base": 500000,
        "rope_scaling_factor": 8,
        "norm_epsilon": 1e-5,
        "normalization": "RMSNorm",
        "padded_vocab_size": 128128,
        "untie_embeddings_and_output_weights": True,
        "max_position_embeddings": 8192,
    }
}

def parse_args():
    # Find and remove the --llama argument before other parsers see it
    llama_size = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--llama' and i + 1 < len(sys.argv):
            llama_size = sys.argv[i + 1]
            # Remove both the flag and its value
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
        i += 1
    
    if llama_size is None:
        raise ValueError("--llama argument is required")
    
    if llama_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Llama model size: {llama_size}. Available options are: {', '.join(MODEL_CONFIGS.keys())}")
    
    return llama_size

def main():
    # Parse custom arguments first and remove them from sys.argv
    llama_size = parse_args()
    
    # Get the model configuration based on the specified Llama model size
    model_config = MODEL_CONFIGS[llama_size]
    
    # Apply ALL model configuration parameters directly 
    args_defaults = {
        "transformer_impl": "transformer_engine",
        "use_checkpoint_args": True,
        "ckpt_format": "torch_dist",
        "ckpt_convert_format": "torch",
        "no_load_rng": True,
        "no_load_optim": True,
        "no_save_optim": True,
        
        # Fake args for initialization
        "micro_batch_size": 1,
        "train_iters": 1,
        "lr": 0.0,
    }
    
    # Add all model config parameters to args_defaults
    for key, value in model_config.items():
        args_defaults[key] = value

    initialize_megatron(
        args_defaults=args_defaults,
    )
    args = get_args()
    assert args.load is not None, "You must specify --load"
    assert args.ckpt_convert_save is not None, "You must specify --ckpt-convert-save"
    setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)

if __name__ == "__main__":
    main()