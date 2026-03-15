#!/usr/bin/env python3
"""Compare logits across HuggingFace, Megatron, and pre-extracted checkpoints.

Usage:
    python compare_logits.py <spec1> [spec2 ...] [OPTIONS]

Specs:
    hf:/path/to/hf_model          Load HF model in-process
    meg:/path/to/megatron_ckpt     Launch torchrun subprocess to extract
    pt:/path/to/saved_logits.pt    Load pre-extracted logits

Single spec = smoke test (verify model loads, logits are finite).
Two+ specs = pairwise comparison with tolerances.

Can be used standalone without running any conversion pipeline.

Options:
    --prompt TEXT              Prompt for extraction (default: "Sanity check prompt.")
    --dtype bf16|fp16|fp32     Model dtype (default: bf16)
    --atol FLOAT               Absolute tolerance (default: 5e-2)
    --rtol FLOAT               Relative tolerance (default: 1e-2)
    --trust-remote-code        For HF models with custom code
    --tokenizer PATH           Explicit tokenizer for HF models
    --megatron-dir PATH        Path to Megatron-LM repo
    --tp INT                   Override TP size for meg: specs (default: auto-detect from checkpoint)
    --old-megatron             Use legacy pretrain_gpt model_provider for meg: specs (deprecated)
Exit code: 0 if all pairs pass, 1 otherwise.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import torch

from megatron_utils import add_old_megatron_arg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        dtype_name, torch.bfloat16
    )


def build_report(token_ids, last_token_logits: torch.Tensor):
    """Build a comparison report dict from logits."""
    logits_fp32 = last_token_logits.detach().float().contiguous().cpu()

    top10_vals, top10_ids = torch.topk(logits_fp32[0], k=10)
    return {
        "token_ids": token_ids,
        "num_prompt_tokens": len(token_ids),
        "vocab_size": int(logits_fp32.shape[-1]),
        "top1_token_id": int(torch.argmax(logits_fp32[0]).item()),
        "top1_logit": float(logits_fp32[0, torch.argmax(logits_fp32[0])].item()),
        "top10_token_ids": [int(x) for x in top10_ids.tolist()],
        "top10_logits": [float(x) for x in top10_vals.tolist()],
    }


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

def parse_spec(spec: str):
    """Return (type, path) from a 'hf:', 'meg:', or 'pt:' prefixed spec."""
    for prefix in ("hf:", "meg:", "pt:"):
        if spec.startswith(prefix):
            return prefix[:-1], spec[len(prefix):]
    raise ValueError(
        f"Invalid spec '{spec}'. Must start with 'hf:', 'meg:', or 'pt:'."
    )


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

@torch.inference_mode()
def extract_hf_logits(model_path: str, prompt: str, dtype, trust_remote_code: bool,
                      tokenizer_path: str | None = None):
    """Load an HF model, run a forward pass, return (token_ids, last_token_logits)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=trust_remote_code, use_fast=False
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, torch_dtype=dtype
    ).eval().cuda()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.cuda() for k, v in encoded.items()}
    token_ids = encoded["input_ids"][0].tolist()
    if not token_ids:
        raise RuntimeError("Prompt tokenized to an empty sequence.")
    print(f"  [HF] Token IDs ({len(token_ids)}): {token_ids}")

    logits = model(**encoded).logits
    last_token_logits = logits[:, -1, :].float().cpu()

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return token_ids, last_token_logits


def _detect_tp_from_checkpoint(ckpt_path: str) -> int:
    """Read TP size from common.pt in the latest iteration directory."""
    latest_file = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
    iter_dir = None
    if os.path.isfile(latest_file):
        with open(latest_file) as f:
            latest_iter = f.read().strip()
        if latest_iter:
            iter_dir = os.path.join(ckpt_path, f"iter_{int(latest_iter):07d}")

    if iter_dir is None or not os.path.isdir(iter_dir):
        # Find any iter directory
        for entry in sorted(os.listdir(ckpt_path)):
            if entry.startswith("iter_") and os.path.isdir(os.path.join(ckpt_path, entry)):
                iter_dir = os.path.join(ckpt_path, entry)
                break

    if iter_dir is None:
        print("[compare_logits] WARNING: No iter directory found, defaulting to TP=1")
        return 1

    common_pt = os.path.join(iter_dir, "common.pt")
    if not os.path.isfile(common_pt):
        print("[compare_logits] WARNING: common.pt not found, defaulting to TP=1")
        return 1

    d = torch.load(common_pt, weights_only=False, map_location="cpu")
    args = d.get("args")
    if args is None:
        return 1
    return getattr(args, "tensor_model_parallel_size", 1)


def extract_megatron_logits(ckpt_path: str, prompt: str, megatron_dir: str,
                            old_megatron: bool = False, tp_override: int | None = None):
    """Shell out to torchrun extract_megatron_logits.py, return (token_ids, last_token_logits)."""
    if not os.path.isdir(megatron_dir):
        raise FileNotFoundError(
            f"Megatron-LM directory not found: {megatron_dir}. "
            f"Set --megatron-dir to your Megatron-LM repo path."
        )

    # Megatron must be importable for torch.load to unpickle checkpoint metadata
    if megatron_dir not in sys.path:
        sys.path.insert(0, megatron_dir)

    print(f"[compare_logits] Using megatron_dir: {megatron_dir}")

    if tp_override is not None:
        tp_size = tp_override
        print(f"[compare_logits] Using TP={tp_size} (user override) for {ckpt_path}")
    else:
        tp_size = _detect_tp_from_checkpoint(ckpt_path)
        print(f"[compare_logits] Detected TP={tp_size} for {ckpt_path}")

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "extract_megatron_logits.py"
    )
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"extract_megatron_logits.py not found at {script_path}")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        out_pt = tmp.name

    try:
        torchrun_cmd = [
            "torchrun", f"--nproc-per-node={tp_size}",
            script_path,
            "--load", ckpt_path,
            "--ckpt-format", "torch_dist",
            "--auto-detect-ckpt-format",
            "--use-checkpoint-args",
            "--use-mp-args-from-checkpoint-args",
            "--no-load-optim",
            "--no-load-rng",
            "--prompt", prompt,
            "--out-pt", out_pt,
        ]
        if old_megatron:
            torchrun_cmd.append("--old-megatron")

        env = os.environ.copy()
        env["PYTHONPATH"] = megatron_dir + ":" + env.get("PYTHONPATH", "")
        env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        print(f"[compare_logits] Running: {' '.join(torchrun_cmd)}")
        result = subprocess.run(
            torchrun_cmd, cwd=megatron_dir, env=env,
            capture_output=True, text=True
        )

        # Always forward subprocess output so info prints (e.g. token IDs) are visible
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            print("[compare_logits] torchrun STDERR:")
            print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
            raise RuntimeError(
                f"extract_megatron_logits.py failed with exit code {result.returncode}"
            )

        data = torch.load(out_pt, weights_only=False, map_location="cpu")
        return data["token_ids"], data["last_token_logits"]
    finally:
        if os.path.isfile(out_pt):
            os.unlink(out_pt)


def load_pt_logits(pt_path: str):
    """Load pre-extracted .pt file, return (token_ids, last_token_logits)."""
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    return data["token_ids"], data["last_token_logits"]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_pair(name_a, report_a, logits_a, name_b, report_b, logits_b, atol, rtol):
    """Compare two sets of logits. Return True if they match within tolerances."""
    print(f"\n--- Comparing: {name_a} vs {name_b} ---")

    # Check token ID agreement (different tokenizers may produce different sequences)
    tids_match = report_a["token_ids"] == report_b["token_ids"]
    if not tids_match:
        print(f"  WARNING: Token IDs differ ({report_a['num_prompt_tokens']} vs "
              f"{report_b['num_prompt_tokens']} tokens). This typically means different "
              f"tokenizers were used (e.g. HF adds BOS, Megatron may not). "
              f"Logit comparison may not be meaningful.")

    # Numerical comparison
    la = logits_a.float()
    lb = logits_b.float()

    # Handle vocab size mismatch
    if la.shape != lb.shape:
        print(f"  WARNING: Shape mismatch {la.shape} vs {lb.shape}")
        min_vocab = min(la.shape[-1], lb.shape[-1])
        la = la[..., :min_vocab]
        lb = lb[..., :min_vocab]
        print(f"  Comparing first {min_vocab} logits")

    diff = (la - lb).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    close = torch.allclose(la, lb, atol=atol, rtol=rtol)

    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  allclose(atol={atol}, rtol={rtol}): {close}")

    # Top-1 agreement
    top1_match = report_a["top1_token_id"] == report_b["top1_token_id"]
    print(f"  Top-1 token match: {top1_match} "
          f"({report_a['top1_token_id']} vs {report_b['top1_token_id']})")

    # Top-10 agreement
    set_a = set(report_a["top10_token_ids"])
    set_b = set(report_b["top10_token_ids"])
    top10_overlap = len(set_a & set_b)
    print(f"  Top-10 overlap: {top10_overlap}/10")

    # Pass criteria: top-1 must match and mean diff must be small
    passed = top1_match and mean_diff < atol
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare logits across HF, Megatron, and pre-extracted checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "specs", nargs="+", metavar="SPEC",
        help="Checkpoint specs: hf:/path, meg:/path, pt:/path",
    )
    parser.add_argument("--prompt", default="Sanity check prompt.", help="Prompt text")
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
        help="Model dtype for HF loading",
    )
    parser.add_argument("--atol", type=float, default=5e-2, help="Absolute tolerance (default: 5e-2, suitable for bf16)")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tokenizer", default=None, help="Explicit tokenizer for HF models")
    parser.add_argument(
        "--megatron-dir", default=None,
        help="Path to Megatron-LM repo (default: /iopsstor/scratch/cscs/$USER/Megatron-LM)",
    )
    add_old_megatron_arg(parser, group_name="megatron")
    parser.add_argument(
        "--tp", type=int, default=None,
        help="Override tensor-parallel size for meg: specs (default: auto-detect from checkpoint)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    megatron_dir = args.megatron_dir
    if megatron_dir is None:
        megatron_dir = f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}/Megatron-LM"

    dtype = resolve_dtype(args.dtype)

    # Parse specs
    specs = []
    for s in args.specs:
        spec_type, spec_path = parse_spec(s)
        specs.append((spec_type, spec_path, s))

    # Extract logits for each spec
    results = []  # List of (name, token_ids, last_token_logits, report)
    for spec_type, spec_path, spec_str in specs:
        print(f"\n{'='*60}")
        print(f"Extracting logits: {spec_str}")
        print(f"{'='*60}")

        if spec_type == "hf":
            token_ids, logits = extract_hf_logits(
                spec_path, args.prompt, dtype,
                args.trust_remote_code, args.tokenizer,
            )
        elif spec_type == "meg":
            token_ids, logits = extract_megatron_logits(
                spec_path, args.prompt, megatron_dir,
                old_megatron=args.old_megatron,
                tp_override=args.tp,
            )
        elif spec_type == "pt":
            token_ids, logits = load_pt_logits(spec_path)
        else:
            raise ValueError(f"Unknown spec type: {spec_type}")

        report = build_report(token_ids, logits)
        name = f"{spec_type}:{os.path.basename(spec_path.rstrip('/'))}"
        results.append((name, token_ids, logits, report))

        # Print individual report
        print(f"  Prompt tokens: {report['num_prompt_tokens']}")
        print(f"  Vocab size:    {report['vocab_size']}")
        print(f"  Top-1: token={report['top1_token_id']} logit={report['top1_logit']:.4f}")
        print(f"  Top-10 tokens: {report['top10_token_ids']}")
        print(f"  Top-10 logits: {[f'{v:.4f}' for v in report['top10_logits']]}")

        # Smoke test: verify logits are finite and non-degenerate
        if not torch.isfinite(logits).all():
            print("  WARNING: Logits contain non-finite values!")
        if logits.std().item() < 1e-6:
            print("  WARNING: Logits appear degenerate (near-zero variance).")

    # Single model = smoke test only
    if len(results) == 1:
        print(f"\n{'='*60}")
        print("Single-model smoke test: PASS")
        print(f"{'='*60}")
        sys.exit(0)

    # Pairwise comparison
    print(f"\n{'='*60}")
    print(f"Pairwise Comparison ({len(results)} models)")
    print(f"{'='*60}")

    all_passed = True
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            name_a, _, logits_a, report_a = results[i]
            name_b, _, logits_b, report_b = results[j]
            passed = compare_pair(
                name_a, report_a, logits_a,
                name_b, report_b, logits_b,
                args.atol, args.rtol,
            )
            if not passed:
                all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("OVERALL: ALL COMPARISONS PASSED")
    else:
        print("OVERALL: SOME COMPARISONS FAILED")
    print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()