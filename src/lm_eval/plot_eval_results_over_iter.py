import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def extract_iteration(path):
    """
    Extract the trailing number from the checkpoint folder name,
    e.g. '...-0.9i-0.1t-2700' -> 2700
    """
    # Handle the initial checkpoint explicitly
    if "Llama-3.2-3B" in path:
        return 0

    p = Path(path)
    # Go two levels up: .../checkpoint_dir/results_xxx.json
    checkpoint_dir = p.parent.parent.name
    match = re.search(r'-(\d+)$', checkpoint_dir)
    if match:
        return int(match.group(1))
    return None

def load_benchmarks(json_file):
    """Load benchmark results from a single JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)

    results = data["results"]

    benchmarks = {
        "arc_challenge": results["arc_challenge"],
        "arc_easy": results["arc_easy"],
        "hellaswag": results["hellaswag"],
        "piqa": results["piqa"],
        "winogrande": results["winogrande"],
    }

    accs = {}
    stderrs = {}
    for name, vals in benchmarks.items():
        acc = vals.get("acc,none")
        stderr = vals.get("acc_stderr,none", 0.0)
        if acc is not None:
            accs[name] = acc
            stderrs[name] = stderr

    return accs, stderrs

def plot_progress(json_files, output_file="/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/plots/training_progress.png"):
    # Sort by iteration number
    parsed = [(extract_iteration(f), f) for f in json_files]
    for i, (it, f) in enumerate(parsed):
        if it is None:
            parsed[i] = (i, f)  # fallback to order
    parsed.sort(key=lambda x: x[0])

    iterations = []
    benchmark_history = {}
    benchmark_stderr = {}

    for it, f in parsed:
        accs, errs = load_benchmarks(f)
        iterations.append(it)
        for name in accs:
            benchmark_history.setdefault(name, []).append(accs[name])
            benchmark_stderr.setdefault(name, []).append(errs[name])

    # Plot
    plt.figure(figsize=(12, 6))
    for name in benchmark_history:
        accs = np.array(benchmark_history[name])
        errs = np.array(benchmark_stderr[name])
        plt.plot(iterations, accs, marker="o", label=name)
        plt.fill_between(iterations, accs - errs, accs + errs, alpha=0.2)

    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.title("Benchmark Accuracy vs Training Iterations")
    plt.legend(title="Benchmark", loc="lower right")  # 👈 place legend bottom right
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"✅ Saved plot to {output_file}")


if __name__ == "__main__":
    files = [
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-2700/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-2700__HF/results_2025-09-15T11-37-46.377134.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-5400/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-5400__HF/results_2025-09-15T11-42-48.980990.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-8100/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-8100__HF/results_2025-09-15T11-46-01.274955.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-10800/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-10800__HF/results_2025-09-15T11-47-26.642186.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-13500/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-13500__HF/results_2025-09-15T11-47-23.053021.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-16200/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-16200__HF/results_2025-09-15T11-48-23.330286.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-18900/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-18900__HF/results_2025-09-15T11-50-03.263526.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-21600/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-21600__HF/results_2025-09-15T11-50-02.933361.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-24300/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama-3b-15n-8192sl-120gbsz-0.9i-0.1t-24300__HF/results_2025-09-15T11-50-02.878278.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-27000/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-27000__HF/results_2025-09-15T09-59-25.371298.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/Llama-3.2-3B/__iopsstor__scratch__cscs__nirmiger__Llama-3.2-3B/results_2025-09-02T14-11-42.014991.json",
    ]
    plot_progress(files, output_file="/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/plots/training_progress_2.png")
