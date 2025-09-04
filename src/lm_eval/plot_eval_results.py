import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_model_name(path):
    """Extract model name from file path (the directory before the __...__ part)."""
    p = Path(path)
    parent = p.parent.name
    # If it starts with __, go one level higher
    if parent.startswith("__"):
        parent = p.parent.parent.name
    return parent


def load_benchmarks(json_file):
    """Load benchmark results from a single JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)

    results = data["results"]
    groups = data["groups"]

    benchmarks = {
        "arc_challenge": results["arc_challenge"],
        "arc_easy": results["arc_easy"],
        "commonsense_qa": results["commonsense_qa"],
        "hellaswag": results["hellaswag"],
        "piqa": results["piqa"],
        "winogrande": results["winogrande"],
    }

    # Add grouped MMLU results
    for k, v in groups.items():
        benchmarks[k] = v

    # Extract labels, accuracies, and errors
    labels, accuracies, errors = [], [], []
    for name, vals in benchmarks.items():
        acc = vals.get("acc,none")
        stderr = vals.get("acc_stderr,none", 0.0)
        if acc is not None:
            labels.append(vals.get("alias", name).strip())
            accuracies.append(acc)
            errors.append(stderr)

    return labels, accuracies, errors


def plot_multiple_models(json_files, output_file="llm_comparison.png"):
    all_labels = None
    all_accuracies = []
    all_errors = []
    model_names = [extract_model_name(f) for f in json_files]

    # Load all models
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
    width = 0.12 if len(model_names) > 1 else 0.3  # dynamic bar width
    plt.figure(figsize=(14, 7))

    for i, (accs, errs) in enumerate(zip(all_accuracies, all_errors)):
        offset = (i - len(all_accuracies) / 2) * width + width / 2
        plt.bar(x + offset, accs, width, yerr=errs, capsize=4, label=model_names[i])

    plt.xticks(x, all_labels, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("LLM Benchmark Comparison")
    plt.legend()
    plt.tight_layout()

    # Save instead of showing
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    files = [
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/Llama-3.2-3B/__iopsstor__scratch__cscs__nirmiger__Llama-3.2-3B/results_2025-09-02T14-11-42.014991.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.4-0.6/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.4-0.6__HF/results_2025-09-04T12-19-35.645805.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-1n-8192sl-64gbsz-test/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-1n-8192sl-64gbsz-test__HF/results_2025-09-02T13-40-15.236029.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.5-0.5/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.5-0.5__HF/results_2025-09-04T12-19-57.320606.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.6-0.4/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.6-0.4__HF/results_2025-09-04T12-20-10.204845.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-0.8-0.2/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-0.8-0.2__HF/results_2025-09-04T12-20-13.740250.json",
        "/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/llama3-3b-2n-8192sl-120gbsz-1.0-0.0/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-2n-8192sl-120gbsz-1.0-0.0__HF/results_2025-09-04T12-20-41.271584.json",
    ]
    plot_multiple_models(files, output_file="comparison.png")
