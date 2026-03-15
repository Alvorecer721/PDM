# Logits Comparison Tool

Verify that converted HuggingFace checkpoints produce matching logits against reference models or Megatron checkpoints.

## Overview

Two scripts work together:

- **`compare_logits.py`** -- Main entry point. Accepts `hf:`, `meg:`, and `pt:` checkpoint specs, extracts logits, and compares them pairwise.
- **`extract_megatron_logits.py`** -- Helper launched via `torchrun` by `compare_logits.py` when processing `meg:` specs. Not called directly.

## Checkpoint Specs

| Prefix | Description | Runs in |
|--------|-------------|---------|
| `hf:/path/to/hf_model` | HuggingFace model directory | Current process (needs `transformers` + GPU) |
| `meg:/path/to/megatron_ckpt` | Megatron `torch_dist` checkpoint root | `torchrun` subprocess (needs Megatron + GPU) |
| `pt:/path/to/saved.pt` | Pre-extracted logits `.pt` file | CPU only, no model loading |

### Megatron checkpoint path structure

The `meg:` path must point to the checkpoint root containing `latest_checkpointed_iteration.txt` and iteration subdirectories:

```
/path/to/megatron_ckpt/
  latest_checkpointed_iteration.txt
  iter_0007000/
    common.pt
    ...
```

TP (tensor parallelism) is auto-detected from `common.pt`.

## Standalone Usage (without conversion)

### Environment requirements

- **For `hf:` specs**: needs `transformers`, `torch`, and a GPU.
- **For `meg:` specs**: needs Megatron-LM on `PYTHONPATH` and GPUs matching the checkpoint's TP size.
- **For `pt:` specs**: CPU-only, no special requirements.

### Running on a compute node (already inside container)

If you are inside an interactive `srun` session or a SLURM job with the right container:

```bash
# Smoke test a single HF model (verify it loads, logits are finite)
python src/convert/compare_logits.py hf:/path/to/hf_model

# Compare two HF models
python src/convert/compare_logits.py hf:/path/to/model_a hf:/path/to/model_b

# Compare HF model against pre-extracted logits
python src/convert/compare_logits.py hf:/path/to/model pt:/path/to/reference.pt

# Compare HF model against Megatron checkpoint
python src/convert/compare_logits.py \
    hf:/path/to/hf_model \
    meg:/path/to/megatron_ckpt \
    --megatron-dir /iopsstor/scratch/cscs/$USER/Megatron-LM
```

### Running from a login node (meg: specs need a container)

When you're on a login node and need to compare against a Megatron checkpoint, use `--container-env` to wrap the `torchrun` subprocess with `srun --environment=<toml>`:

```bash
python src/convert/compare_logits.py \
    hf:/path/to/hf_model \
    meg:/path/to/megatron_ckpt \
    --container-env /users/rkreft/.edf/new_ngc.toml \
    --megatron-dir /iopsstor/scratch/cscs/$USER/Megatron-LM
```

Note: `hf:` specs always run in the current process, so `transformers` must be available in your current Python environment regardless of `--container-env`.

### All CLI options

```
python src/convert/compare_logits.py <spec1> [spec2 ...] [OPTIONS]

Options:
  --prompt TEXT              Prompt for logit extraction (default: "Sanity check prompt.")
  --dtype bf16|fp16|fp32     Model dtype for HF loading (default: bf16)
  --atol FLOAT               Absolute tolerance (default: 1e-2)
  --rtol FLOAT               Relative tolerance (default: 1e-2)
  --trust-remote-code        Pass trust_remote_code=True to HF loaders
  --tokenizer PATH           Explicit tokenizer for HF models (default: model dir)
  --megatron-dir PATH        Path to Megatron-LM repo
                             (default: /iopsstor/scratch/cscs/$USER/Megatron-LM)
  --old-megatron          Use old pretrain_gpt.model_provider for meg: specs
                             (deprecated, for old Megatron API checkpoints)
  --container-env PATH       Container .toml for meg: specs. Wraps torchrun with
                             srun --environment=PATH. Not needed if already inside container.
```

### Exit codes

- `0` -- All pairwise comparisons passed (allclose + top-1 token match), or single-model smoke test passed.
- `1` -- At least one comparison failed.

## Integrated with Conversion Pipelines

### Apertus

```bash
# Via SLURM -- conversion + logits smoke test of the converted HF model (new Megatron API, default)
sbatch submissions/submit-convert.slurm \
    /path/to/experiment \
    --model-type apertus \
    --checkpoint-path /path/to/megatron_ckpt \
    --logits-test

# With a reference HF model for comparison
sbatch submissions/submit-convert.slurm \
    /path/to/experiment \
    --model-type apertus \
    --checkpoint-path /path/to/megatron_ckpt \
    --logits-test \
    --logits-test-ref hf:/path/to/reference_hf_model

# Legacy Megatron API (deprecated)
sbatch submissions/submit-convert.slurm \
    /path/to/experiment \
    --model-type apertus \
    --checkpoint-path /path/to/megatron_ckpt \
    --old-megatron \
    --logits-test
```

### Llama3

```bash
sbatch submissions/submit-convert.slurm \
    /path/to/experiment \
    --model-type llama3 \
    --logits-test
```

### Re-running logits test on already-converted checkpoints

The conversion scripts skip conversion if the HF directory already exists, but will still run the logits test when `--logits-test` is passed. So you can re-submit the same job to run just the logits comparison:

```bash
# HF/ already exists, only the logits test runs
sbatch submissions/submit-convert.slurm \
    /path/to/experiment \
    --model-type apertus \
    --checkpoint-path /path/to/megatron_ckpt \
    --logits-test \
    --logits-test-ref hf:/path/to/reference
```

## Notes

- **HF vs Megatron tokenizer mismatch**: When comparing `hf:` vs `meg:` specs, be aware that HF tokenizers typically add BOS tokens while Megatron's `tokenizer.tokenize()` may not. The tool warns when token IDs differ between specs. For reliable cross-format comparison, use the same prompt and verify the token count in the output.
- **GPU memory**: HF models are unloaded after logit extraction to free GPU memory before processing the next spec.
- **The logits test is non-blocking**: When run as part of conversion, a logits test failure prints a warning but does not fail the overall conversion job.
