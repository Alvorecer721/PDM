#!/usr/bin/env python3
"""Extract last-token logits from a Megatron torch_dist checkpoint.

Must be launched via torchrun:
    torchrun --nproc-per-node=<TP> extract_megatron_logits.py \
        --load <ckpt_path> --prompt "..." --out-pt <output.pt> \
        --ckpt-format torch_dist --auto-detect-ckpt-format \
        --use-checkpoint-args --use-mp-args-from-checkpoint-args \
        --no-load-optim --no-load-rng

Supports both new Megatron API (model_provider + gpt_builder, default) and
legacy Megatron API (pretrain_gpt.model_provider, via --old-megatron).
The --old-megatron flag is deprecated and will be removed in a future version.
"""
import os

import torch

from megatron.core import mpu
from megatron_utils import add_old_megatron_arg, get_model_provider_fn
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.utils import get_ltor_masks_and_position_ids


def patch_te_set_extra_state_eof():
    """Ignore corrupt/empty TE extra_state blobs when loading older checkpoints."""
    try:
        from transformer_engine.pytorch.module import base as te_base
    except Exception:
        return

    cls = te_base.TransformerEngineBaseModule
    if getattr(cls, "_patched_ignore_eof_extra_state", False):
        return

    original_set_extra_state = cls.set_extra_state

    def safe_set_extra_state(self, state):
        try:
            return original_set_extra_state(self, state)
        except EOFError:
            return None

    cls.set_extra_state = safe_set_extra_state
    cls._patched_ignore_eof_extra_state = True


def extra_args(parser):
    group = parser.add_argument_group("logits-extraction")
    group.add_argument("--prompt", type=str, required=True)
    group.add_argument("--out-pt", type=str, required=True, help="Output .pt file path")
    add_old_megatron_arg(parser, group_name="logits-extraction")
    return parser


@torch.inference_mode()
def main():
    initialize_megatron(
        extra_args_provider=extra_args,
        args_defaults={
            "use_checkpoint_args": True,
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
            "bf16": True, # use bf16 inference as hf is inference in bf16 as well
        },
    )

    args = get_args()
    if args.pipeline_model_parallel_size != 1:
        raise RuntimeError(
            f"This script assumes PP=1, got PP={args.pipeline_model_parallel_size}."
        )

    patch_te_set_extra_state_eof()

    provider_fn = get_model_provider_fn(args.old_megatron)
    model = get_model(provider_fn, wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=True)
    model = model[0]
    model.eval()

    tokenizer = get_tokenizer()
    token_ids = [int(x) for x in tokenizer.tokenize(args.prompt)]
    if len(token_ids) == 0:
        raise RuntimeError("Prompt tokenized to an empty sequence.")
    if mpu.get_tensor_model_parallel_rank() == 0:
        print(f"  [Megatron] Token IDs ({len(token_ids)}): {token_ids}")

    tokens = torch.tensor(token_ids, dtype=torch.long, device="cuda").unsqueeze(0)

    eod_token = getattr(tokenizer, "eod", None)
    if eod_token is None:
        eod_token = getattr(tokenizer, "eos", 0)

    # Build kwargs compatible with both old and new Megatron signatures.
    # Old Megatron: (data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss)
    # New Megatron adds: pad_token, pad_mask_loss
    import inspect
    sig_params = inspect.signature(get_ltor_masks_and_position_ids).parameters
    mask_kwargs = dict(
        data=tokens,
        eod_token=eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    if "pad_token" in sig_params:
        pad_token = getattr(tokenizer, "pad", 0)
        mask_kwargs["pad_token"] = pad_token
    if "pad_mask_loss" in sig_params:
        mask_kwargs["pad_mask_loss"] = False

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(**mask_kwargs)

    logits = model(tokens, position_ids, attention_mask)
    if args.tensor_model_parallel_size > 1:
        logits = gather_from_tensor_model_parallel_region(logits)
    last_token_logits = logits[:, -1, :].float().cpu()

    if mpu.get_tensor_model_parallel_rank() == 0:
        out_dir = os.path.dirname(args.out_pt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        torch.save(
            {
                "prompt": args.prompt,
                "token_ids": token_ids,
                "last_token_logits": last_token_logits,
            },
            args.out_pt,
        )
        print(f"[extract_megatron_logits] Saved to {args.out_pt} shape={tuple(last_token_logits.shape)}")


if __name__ == "__main__":
    main()