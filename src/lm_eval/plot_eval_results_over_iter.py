import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import seaborn as sns

def collect_json_files(folders):
    """Given a list of folders, collect all JSON result files inside them."""
    json_files = []
    for folder in folders:
        folder = Path(folder)
        if folder.is_file() and folder.suffix == ".json":
            # Direct JSON file path
            json_files.append(folder)
        elif folder.is_dir():
            # Recursively look for results_*.json
            json_files.extend(folder.rglob("results_*.json"))
        else:
            print(f"⚠️ Skipping invalid path: {folder}")
    return [str(f) for f in sorted(json_files)]


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

def compute_iterations(json_files):
    """Return iterations (in billions of tokens) for all files, 
    continuing across multiple runs."""
    # Split files into old run vs paired run
    old_run = [f for f in json_files if "paired" not in f]
    new_run = [f for f in json_files if "paired" in f]

    # Parse iterations
    parsed_old = [(extract_iteration(f), f) for f in old_run]
    parsed_new = [(extract_iteration(f), f) for f in new_run]

    # Fallback for None
    for i, (it, f) in enumerate(parsed_old):
        if it is None:
            parsed_old[i] = (i, f)
    for i, (it, f) in enumerate(parsed_new):
        if it is None:
            parsed_new[i] = (i, f)

    # Sort each run
    parsed_old.sort(key=lambda x: x[0])
    parsed_new.sort(key=lambda x: x[0])

    # Compute token scaling (Attention: This is hardcoded for specific runs)
    print("Assuming image token ratio of 0.8.")
    scale = 8192 * 120 * 0.9 / 1e9

    # Old run iterations
    iterations_old = [it * scale for it, _ in parsed_old]

    # Offset = last token count of old run
    offset = iterations_old[-1] if iterations_old else 0.0

    # New run iterations (continued)
    iterations_new = [offset + it * scale for it, _ in parsed_new]

    # Merge runs
    parsed = parsed_old + parsed_new
    iterations = iterations_old + iterations_new

    return parsed, iterations


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

def plot_progress(json_files, output_file="/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/plots/training_progress.pdf"):
    # Set up LaTeX fonts and styling
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    sns.set_theme(style="whitegrid", font="serif", font_scale=2)

    plt.rcParams.update({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.figsize": (9, 8),
        "lines.linewidth": 2,
        "axes.linewidth": 1.0,
        "legend.title_fontsize": 18,
        "axes.titlesize": 22,
        "axes.titleweight": 'bold',
        "font.family": "serif",
    })

    parsed, iterations = compute_iterations(json_files)

    benchmark_history = {}
    benchmark_stderr = {}
    for (it, f), x in zip(parsed, iterations):
        accs, errs = load_benchmarks(f)
        for name in accs:
            benchmark_history.setdefault(name, []).append(accs[name])
            benchmark_stderr.setdefault(name, []).append(errs[name])

    colors = sns.color_palette("colorblind", 5)

    plt.figure(figsize=(9, 8))
    for i, name in enumerate(benchmark_history):
        accs = np.array(benchmark_history[name])
        errs = np.array(benchmark_stderr[name])
        plt.plot(iterations, accs, marker="o", label=name, linewidth=2, markersize=6, color=colors[i])
        plt.fill_between(iterations, accs - errs, accs + errs, alpha=0.2, color=colors[i])

    # === Add Stage switch marker ===
    # Last iteration from Stage 1 = last "long-run" file
    stage1_iters = [x for (it, f), x in zip(parsed, iterations) if "paired" not in f]
    if stage1_iters:
        switch_x = stage1_iters[-1]

        # Vertical line
        plt.axvline(x=switch_x, color="black", linestyle="--", linewidth=2)

        # Labels
        ymin, ymax = plt.ylim()
        ymid = (ymin + ymax) / 2
        plt.text(switch_x * 0.5, ymid, r"Stage 1", ha="center", va="center", fontsize=20)
        plt.text(switch_x + (plt.xlim()[1] - switch_x) / 2, ymid, r"Stage 2", ha="center", va="center", fontsize=20)

    # Labels
    plt.xlabel(r"Consumed Image Tokens (B)", fontsize=20, weight="bold")
    plt.ylabel(r"Accuracy", fontsize=20, weight="bold")

    plt.legend(
        title=r"\textbf{Benchmark}",
        loc="lower right",
        fontsize=16,
        title_fontsize=18,
        frameon=True,
        framealpha=0.7,
        ncol=1,
        borderpad=1,
        labelspacing=0.5,
        handlelength=2,
        columnspacing=1
    )

    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(left=0, right=80)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format='pdf')
    plt.close()

    print(f"✅ Saved plot to {output_file}")


if __name__ == "__main__":
    folders = [
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0006350",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0012700",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0019050",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0025400",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0031750",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0038100",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0044450",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0050800",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0057150",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-long-run-0063500",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0002270",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0004540",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0006810",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0009080",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0011350",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0013620",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0015890",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0018160",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0020430",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0022700",
        "/Users/nicolairmiger/PDM/results/lm_eval/Llama-3.2-3B",
    ]
    files = collect_json_files(folders)
    plot_progress(files, output_file="/Users/nicolairmiger/PDM/results/lm_eval/plots/training_progress_2.pdf")
