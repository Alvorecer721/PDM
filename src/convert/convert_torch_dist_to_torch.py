"""
Convert Megatron torch_dist checkpoint to torch format.

Supports both new Megatron API (model_provider + gpt_builder, default) and
legacy Megatron API (pretrain_gpt.model_provider, via --old-megatron).

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun src/convert/convert_torch_dist_to_torch.py \
    --bf16 \
    --load <checkpoint_path> \
    --ckpt-convert-save <output_path>

# Legacy mode (deprecated):
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun src/convert/convert_torch_dist_to_torch.py \
    --bf16 \
    --old-megatron \
    --load <checkpoint_path> \
    --ckpt-convert-save <output_path>
"""
from megatron.core.enums import ModelType
from megatron.training.training import setup_model_and_optimizer
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
from megatron_utils import add_old_megatron_arg, get_model_provider_fn


def main():

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

    initialize_megatron(
        extra_args_provider=add_old_megatron_arg,
        args_defaults=args_defaults,
    )
    args = get_args()
    assert args.load is not None, "You must specify --load"
    assert args.ckpt_convert_save is not None, "You must specify --ckpt-convert-save"

    provider_fn = get_model_provider_fn(args.old_megatron)
    setup_model_and_optimizer(provider_fn, ModelType.encoder_or_decoder)

if __name__ == "__main__":
    main()
