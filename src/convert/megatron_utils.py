"""Shared utilities for Megatron checkpoint conversion and logits extraction."""
import warnings


def get_model_provider_fn(old_megatron: bool):
    """Return the appropriate model_provider callable.

    Args:
        old_megatron: If True, use legacy pretrain_gpt.model_provider (deprecated).
                      If False (default), use new model_provider(gpt_builder) API.
    """
    if old_megatron:
        warnings.warn(
            "--old-megatron is deprecated. Migrate to the new Megatron API "
            "(model_provider + gpt_builder). This flag will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        from pretrain_gpt import model_provider
        return model_provider
    else:
        from functools import partial
        from gpt_builders import gpt_builder
        from model_provider import model_provider
        return partial(model_provider, gpt_builder)


def add_old_megatron_arg(parser, group_name="conversion"):
    """Add the --old-megatron argument to an argparse parser group."""
    group = parser.add_argument_group(group_name)
    group.add_argument(
        "--old-megatron",
        action="store_true",
        help="(Deprecated) Use legacy pretrain_gpt.model_provider instead of new "
        "model_provider(gpt_builder) API. Will be removed in a future version.",
    )
    return parser
