"""Launcher wrapper that disables wandb on non-main processes.

When using accelerate launch with multiple GPUs, each process initializes
wandb independently, creating duplicate runs. This wrapper disables wandb
on all non-rank-0 processes.

Usage:
    accelerate launch wandb_guard_launcher.py -m lm_eval [args...]
    accelerate launch wandb_guard_launcher.py -m lmms_eval [args...]
"""
import os
import sys
import runpy

if os.environ.get("LOCAL_RANK", "0") != "0":
    os.environ["WANDB_MODE"] = "disabled"

if "-m" in sys.argv:
    idx = sys.argv.index("-m")
    module = sys.argv[idx + 1]
    sys.argv = [module] + sys.argv[idx + 2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    print("Error: -m <module> required", file=sys.stderr)
    sys.exit(1)
