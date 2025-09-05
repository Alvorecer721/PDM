import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def extract_model_name(path):
    """Extract a short model name from the path."""
    p = Path(path)
    parent = p.parent.name

    if parent.startswith("__"):
        parent = p.parent.parent.name

    # Try to find pattern like 0.4-0.6 or 1.0-0.0
    match = re.search(r'(\d\.\d{1,2})-(\d\.\d{1,2})', parent)
    if match:
        return match.group(0)  # e.g., "0.4-0.6"
    else:
        return parent  # fallback

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

    labels, accuracies, errors = [], [], []
    for name, vals in benchmarks.items():
        acc = vals.get("acc,none")
        stderr = vals.get("acc_stderr,none", 0.0)
        if acc is not None:
            labels.append(vals.get("alias", name).strip())
            accuracies.append(acc)
            errors.append(stderr)

    return labels, accuracies, errors

def sort_files_by_config(files):
    def extract_ratio_key(path):
        match = re.search(r'(\d\.\d{1,2})-(\d\.\d{1,2})', path)
        if match:
            alpha = float(match.group(1))
            beta = float(match.group(2))
            return (1, alpha, beta)  # group 1 = "has alpha-beta"
        else:
            return (0, 0.0, 0.0)  # group 0 = "fallback", goes first

    return sorted(files, key=extract_ratio_key)


def plot_multiple_models(json_files, output_file="llm_comparison.png"):
    json_files = sort_files_by_config(json_files)

    all_labels = None
    all_accuracies = []
    all_errors = []
    model_names = [extract_model_name(f) for f in json_files]

    for jf in json_files:
        labels, accs, errs = load_benchmarks(jf)
        if all_labels is None:
            all_labels = labels
        else:
            assert labels == all_labels, "Labels must match across models"
        all_accuracies.append(accs)
        all_errors.append(errs)

    # Plot grouped bars
    x = np.arange(len(all_labels))
    num_models = len(model_names)
    width = 0.8 / num_models  # distribute total width of 0.8

    plt.figure(figsize=(16, 8))

    for i, (accs, errs) in enumerate(zip(all_accuracies, all_errors)):
        offset = (i - num_models / 2) * width + width / 2
        plt.bar(x + offset, accs, width, yerr=errs, capsize=4, label=model_names[i])

    plt.xticks(x, all_labels, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("LLM Benchmark Comparison (1B image tokens)")
    plt.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✅ Plot saved to {output_file}")


if __name__ == "__main__":
    files = [
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/Llama-3.2-3B/__iopsstor__scratch__cscs__nirmiger__Llama-3.2-3B/results_2025-09-02T14-11-42.014991.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.4-0.6/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.4-0.6__HF/results_2025-09-04T12-19-35.645805.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.5-0.5/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.5-0.5__HF/results_2025-09-04T12-19-57.320606.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.6-0.4/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.6-0.4__HF/results_2025-09-04T12-20-10.204845.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.8-0.2/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.8-0.2__HF/results_2025-09-04T12-20-13.740250.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-1.0-0.0/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-1.0-0.0__HF/results_2025-09-04T12-20-41.271584.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.0-1.0/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.0-1.0__HF/results_2025-09-05T11-02-00.921162.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.9-0.1/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.9-0.1__HF/results_2025-09-05T11-02-41.151412.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.95-0.05/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.95-0.05__HF/results_2025-09-05T11-02-09.866610.json",
    ]
    plot_multiple_models(files, output_file="comparison.png")
