"""
plot_eval_results.py:

generate bar-plot comparing different models on different benchmarks.
Specify the path to respective json_files (results of PDM eval) down below and run the script like this:
    python -m lm_eval.plot_eval_results --output-path=[PATH_TO_RES_FILE] [--use-latex-text-renderer] [--title="Custom Title"] [--no-sort] [--format=png] [--instruct-bench]

You can specify files in two ways:
    1. Just file paths: The model name will be extracted automatically from the path
    2. Tuples of (file_path, display_name): Use custom display names in the legend

By default, files are sorted by their ratio pattern (e.g., 0.6i-0.4t). Use --no-sort to preserve the order as specified.
Output format defaults to PDF. Use --format to specify png, jpg, svg, or other matplotlib-supported formats.

Benchmark sets:
    - Default: arc_challenge, arc_easy, hellaswag, piqa, winogrande
    - Instruct (--instruct-bench): bbh, mmlu, hellaswag, gsm8k, truthfulqa_mc2, ifeval
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import re

RATIO_PATTERN = re.compile(r'(\d\.\d{1,2})(?:i)?-(\d\.\d{1,2})(?:t)?')

# Benchmark configurations: (benchmark_key, metric_key, stderr_key)
DEFAULT_BENCHMARKS = {
    "arc_challenge": ("acc,none", "acc_stderr,none"),
    "arc_easy": ("acc,none", "acc_stderr,none"),
    "hellaswag": ("acc,none", "acc_stderr,none"),
    "piqa": ("acc,none", "acc_stderr,none"),
    "winogrande": ("acc,none", "acc_stderr,none"),
}

INSTRUCT_BENCHMARKS = {
    "bbh": ("exact_match,get-answer", "exact_match_stderr,get-answer"),
    "mmlu": ("acc,none", "acc_stderr,none"),
    "hellaswag": ("acc,none", "acc_stderr,none"),
    "gsm8k": ("exact_match,strict-match", "exact_match_stderr,strict-match"),
    "truthfulqa_mc2": ("acc,none", "acc_stderr,none"),
    "ifeval": ("prompt_level_strict_acc,none", "prompt_level_strict_acc_stderr,none"),
}

MULTIMODAL_BENCHMARKS = {
    'ai2d': ('exact_match,flexible-extract', 'exact_match_stderr,flexible-extract'),
}

def extract_model_name(path_or_tuple):
    """
    Extract a short model name from the path or tuple.

    Args:
        path_or_tuple: Either a file path string, or a tuple of (file_path, display_name)

    Returns:
        tuple: (file_path, display_name)
    """
    # Check if it's a tuple with a custom name
    if isinstance(path_or_tuple, tuple):
        if len(path_or_tuple) == 2:
            return path_or_tuple[0], path_or_tuple[1]
        else:
            raise ValueError(f"Tuple must have exactly 2 elements (path, name), got {len(path_or_tuple)}")

    # Otherwise, extract name from path
    path = path_or_tuple
    p = Path(path)
    parent = p.parent.name

    if parent.startswith("__"):
        parent = p.parent.parent.name

    # Try to find pattern like 0.4-0.6 or 0.6i-0.4t
    match = RATIO_PATTERN.search(parent)
    if match:
        x, y = match.groups()
        x = float(x)*100
        y = float(y)*100
        if 'i' in parent or 't' in parent:
            display_name = f"{x}% img - {y}% txt"
        else:
            display_name = f"{x}-{y}"
    else:
        display_name = parent  # fallback if no match

    return path, display_name

def load_benchmarks(json_file, benchmark_config=None):
    """Load benchmark results from a single JSON file.

    Args:
        json_file: Path to the JSON results file
        benchmark_config: Dict mapping benchmark names to (metric_key, stderr_key) tuples.
                         If None, uses DEFAULT_BENCHMARKS.

    Returns:
        tuple: (labels, accuracies, errors) lists
    """
    if benchmark_config is None:
        benchmark_config = DEFAULT_BENCHMARKS

    with open(json_file, "r") as f:
        data = json.load(f)

    results = data["results"]

    labels, accuracies, errors = [], [], []
    for name, (metric_key, stderr_key) in benchmark_config.items():
        if name not in results:
            print(f"Warning: Benchmark '{name}' not found in {json_file}, skipping.")
            continue

        vals = results[name]
        acc = vals.get(metric_key)
        stderr = vals.get(stderr_key, 0.0)

        # Handle "N/A" stderr values
        if stderr == "N/A":
            stderr = 0.0

        if acc is not None:
            labels.append(vals.get("alias", name).strip())
            accuracies.append(acc)
            errors.append(stderr)

    return labels, accuracies, errors

def sort_files_by_config(files):
    def extract_ratio_key(path_or_tuple):
        # Extract path from tuple if needed
        if isinstance(path_or_tuple, tuple):
            path = path_or_tuple[0]
        else:
            path = path_or_tuple

        match = RATIO_PATTERN.search(path)
        if match:
            alpha = float(match.group(1))
            beta = float(match.group(2))
            return (1, alpha, beta)  # group 1 = "has alpha-beta"
        else:
            return (0, 0.0, 0.0)  # group 0 = "fallback", goes first

    return sorted(files, key=extract_ratio_key)


def plot_multiple_models(json_files: list, output_file: str, use_tex_text_renderer: bool = False, title: str = None, sort_files: bool = True, output_format: str = 'pdf', benchmark_config=None):
    """
    Plot benchmark results for multiple models.

    Args:
        json_files: List of file paths (str) or tuples of (file_path, display_name)
        output_file: Path to save the output plot
        use_tex_text_renderer: Whether to use LaTeX for text rendering
        title: Custom title for the plot (default: None, no title shown)
        sort_files: Whether to sort files by their ratio pattern (default: True)
        output_format: Output file format (default: 'pdf'). Supports 'png', 'jpg', 'svg', etc.
        benchmark_config: Dict mapping benchmark names to (metric_key, stderr_key) tuples.
                         If None, uses DEFAULT_BENCHMARKS.
    """
    if use_tex_text_renderer:
        # Set up LaTeX fonts and styling
        plt.rc('text', usetex=True)  # Enable LaTeX rendering for text
        plt.rc('font', family='serif')  # Set font to serif (LaTeX default)

    sns.set_theme(style="whitegrid", font="serif", font_scale=1.5)  # Increased font scale for larger text

    plt.rcParams.update({
        "axes.labelsize": 16,  # Increased axis label
        # size
        "xtick.labelsize": 14,  # Increased X tick label size
        "ytick.labelsize": 14,  # Increased Y tick label size
        "legend.fontsize": 12,  # Increased legend font size
        "figure.figsize": (12, 6),  # Make the plot narrower (increase height if needed)
        "lines.linewidth": 2,  # Lines thickness
        "axes.linewidth": 1.0,  # Axis line thickness
        "legend.title_fontsize": 14,  # Increased legend title size
        "axes.titlesize": 18,  # Increased title size
        "axes.titleweight": 'bold',  # Bold title
        "font.family": "serif",  # Ensures consistent font across plots
    })

    # Sort the files by their config if requested
    if sort_files:
        json_files = sort_files_by_config(json_files)

    # Prepare data for plotting - extract paths and names
    all_labels = None
    all_accuracies = []
    all_errors = []
    file_paths = []
    model_names = []

    for f in json_files:
        path, name = extract_model_name(f)
        file_paths.append(path)
        model_names.append(name)

    for jf in file_paths:
        labels, accs, errs = load_benchmarks(jf, benchmark_config)
        if all_labels is None:
            all_labels = labels
        else:
            assert labels == all_labels, "Labels must match across models"
        all_accuracies.append(accs)
        all_errors.append(errs)

    # Create the x locations for each benchmark
    x = np.arange(len(all_labels))
    num_models = len(model_names)
    width = 0.8 / num_models  # Adjust bar width based on the number of models

    # Plot the bars
    plt.figure(figsize=(12, 6))  # Larger figure size for better readability

    # Use colorblind-friendly colors
    colors = sns.color_palette("colorblind", num_models)

    # Plot each model's bars with error bars
    for i, (accs, errs) in enumerate(zip(all_accuracies, all_errors)):
        offset = (i - num_models / 2) * width + width / 2
        plt.bar(x + offset, accs, width, yerr=errs, capsize=4, label=model_names[i], color=colors[i])

    # Adjust axis labels and ticks
    plt.xticks(x, all_labels, rotation=0, ha="center", fontsize=14)
    plt.ylabel(r"Accuracy", weight="bold", fontsize=16)

    # Add title if provided
    if title:
        plt.title(title, fontsize=18, weight='bold', pad=20)

    # Add gridlines and increase the prominence of the axes
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Light grid lines for better readability

    # Place the legend below the plot, outside the chart area
    # Use LaTeX formatting for legend title only if LaTeX renderer is active
    legend_title = r"\textbf{Models}" if use_tex_text_renderer else "Models"

    # Determine number of columns based on number of models (max 4 columns)
    ncol = min(num_models, 4)

    plt.legend(
        title=legend_title,
        loc='upper center',  # Anchor point
        bbox_to_anchor=(0.5, -0.15),  # Place below the x-axis
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
        ncol=ncol,  # Multiple columns to save vertical space
        borderpad=1,
        labelspacing=0.5,
        handlelength=2,
        columnspacing=1.5,
        fancybox=True,
        shadow=True
    )

    # Ensure the layout is tight and the plot is clean
    # Add extra space at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25 + (0.03 * ((num_models - 1) // ncol)))

    # Save the figure in the specified format
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format=output_format)
    plt.close()

    print(f"✅ Plot saved to {output_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--use-latex-text-renderer", action="store_true", help="Use LaTeX to render text in plots. Only works with local tex installation")
    argparser.add_argument("--output-path", type=str, default="test.pdf", help="Path to save the plot")
    argparser.add_argument("--title", type=str, default=None, help="Custom title for the plot (optional)")
    argparser.add_argument("--no-sort", action="store_true", help="Disable automatic sorting of files by ratio pattern (preserves input order)")
    argparser.add_argument("--format", type=str, default="pdf", help="Output file format (default: pdf). Supports png, jpg, svg, etc.")
    argparser.add_argument("--instruct-bench", action="store_true", help="Use instruct benchmarks (bbh, mmlu, hellaswag, gsm8k, truthfulqa_mc2, ifeval) instead of default benchmarks")
    argparser.add_argument("--multi-modal-bench", action="store_true", help="Use multi-modal benchmarks (ai2d, etc.) instead of default benchmarks")
    args = argparser.parse_args()

    # Only use default benchmarks if no flag is given.
    benchmark_config = DEFAULT_BENCHMARKS if not (args.instruct_bench, args.multi_modal_bench) else {}

    # Merge with multi-modal benchmarks if specified
    if args.multi_modal_bench:
        benchmark_config = {**benchmark_config, **MULTIMODAL_BENCHMARKS}

    # Merge with instruct benchmarks if specified
    if args.instruct_bench:
        benchmark_config = {**benchmark_config, **INSTRUCT_BENCHMARKS}

    # Specify the paths to the JSON files that contain the benchmark results
    # You can use either:
    # 1. Plain file paths (names will be auto-extracted):
    files = [
        # Base-Model
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0022700/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-paired-0022700__HF/results_2025-09-20T20-17-47.711456.json", "base-model"),
        # non-packed
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1__HF/results_2025-10-20T17-57-51.888868.json", "PLW0.1"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2__HF/results_2025-10-21T10-28-22.423863.json", "PLW0.2"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3__HF/results_2025-10-20T17-58-01.747630.json", "PLW0.3"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4__HF/results_2025-10-21T10-03-39.507673.json", "PLW0.4"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-FIXES-RPAD/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-FIXES-RPAD__HF/results_2025-10-20T17-43-05.318718.json", "PLW1.0"),
        
        # packed plw confs
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-23T13-26-46.692574.json", "PLW0.1-Packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2-PACKED__HF/results_2025-10-23T13-27-03.584325.json", "PLW0.2-Packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3-PACKED__HF/results_2025-10-24T16-34-15.613037.json", "PLW0.3-Packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4-PACKED__HF/results_2025-10-23T13-27-25.966743.json", "PLW0.4-Packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.5-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.5-PACKED__HF/results_2025-10-23T13-27-37.564768.json", "PLW0.5-Packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-PACKED__HF/results_2025-10-23T13-29-01.138137.json", "PLW1.0-Packed"),
        
        # Different seeds
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-23T13-26-46.692574.json", "Seed-28"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-t-stage2-base-ST-MASKED-PLW0.1-PACKED-SEED42/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-t-stage2-base-ST-MASKED-PLW0.1-PACKED-SEED42__HF/results_2025-10-27T09-33-48.790959.json", "Seed-42"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-t-stage2-base-ST-MASKED-PLW0.1-PACKED-SEED3298/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-t-stage2-base-ST-MASKED-PLW0.1-PACKED-SEED3298__HF/results_2025-10-27T09-34-22.960774.json", "Seed-3298"),
        
        # Different txt/img ratios
        #"/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-23T13-26-46.692574.json",
        #"/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.95i-0.05t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-0.95i-0.05t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-27T11-40-43.573701.json",
        #"/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-27T09-28-01.380367.json",
        #"/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.85i-0.15t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-0.85i-0.15t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-27T17-32-55.441214.json"

        ## Instruct benchmarks ##

        # Different txt/img ratios
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-31T21-51-09.526498.json", "0.9i-0.1t-plw0.1-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.95i-0.05t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-0.95i-0.05t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-31T21-55-18.338526.json", "0.95i-0.05t-plw0.1-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.85i-0.15t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-0.85i-0.15t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-31T21-56-28.163100.json", "0.85i-0.15t-plw0.1-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-31T22-14-49.090708.json", "1.0i-0.0t-plw0.1-packed"),

        # Different PLW values (all only img sft - packed)
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/results_2025-10-31T22-14-49.090708.json", "plw0.1-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2-PACKED__HF/results_2025-10-31T22-06-45.945390.json", "plw0.2-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3-PACKED__HF/results_2025-10-31T22-04-30.293906.json", "plw0.3-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4-PACKED__HF/results_2025-10-31T18-31-22.531173.json", "plw0.4-packed"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.5-PACKED/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.5-PACKED__HF/results_2025-10-31T22-07-06.185690.json", "plw0.5-packed"),
        
         # Different PLW values (all only img sft - non-packed)
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1__HF/results_2025-11-03T11-43-02.387805.json", "plw0.1"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2__HF/results_2025-10-31T22-12-46.392290.json", "plw0.2"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3__HF/results_2025-10-31T22-10-10.012631.json", "plw0.3"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4__HF/results_2025-10-31T22-11-38.505649.json", "plw0.4"),
        #("/users/rkreft/PDM/results/lm_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-FIXES-RPAD/__users__rkreft__megatron-repo__logs__Meg-Runs__image-extension__llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-FIXES-RPAD__HF/results_2025-10-31T22-17-50.278613.json", "plw1.0"),
    
        # LLaMa3.2-3B-Instruct baseline
        #("/users/rkreft/PDM/results/lm_eval/meta-llama__Llama-3.2-3B-Instruct/meta-llama__Llama-3.2-3B-Instruct/results_2025-11-03T13-01-46.805749.json", "LLaMA3.2-3B-Instruct"),

        ## Multi-modal benchmarks ##
        #("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-0.9i-0.1t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/20251104_062906_results.json", "plw0.1-packed-0.9i-0.1t"),
        #("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.85i-0.15t-stage2-base-ST-MASKED-PLW0.1-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-0.85i-0.15t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/20251104_062907_results.json", "plw0.1-packed-0.85i-0.15t"),
        #("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-0.95i-0.05t-stage2-base-ST-MASKED-PLW0.1-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-0.95i-0.05t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/20251104_062909_results.json", "plw0.1-packed-0.95i-0.05t"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1-PACKED__HF/20251104_062915_results.json", "plw0.1-packed-1.0i-0.0t"),

        # Plw variants
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.1__HF/20251104_062914_results.json", "plw0.1"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2__HF/20251104_062911_results.json", "plw0.2"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3__HF/20251104_062908_results.json", "plw0.3"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4__HF/20251104_062912_results.json", "plw0.4"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-FIXES-RPAD/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-USR-MASKED-FIXES-RPAD__HF/20251104_063048_results.json", "plw1.0"),

        # Plw-Packed variants
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.2-PACKED__HF/20251104_062910_results.json", "plw0.2-packed"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.3-PACKED__HF/20251104_062915_results.json", "plw0.3-packed"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.4-PACKED__HF/20251104_062905_results.json", "plw0.4-packed"),
        ("/users/rkreft/PDM/results/lmms_eval/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.5-PACKED/llama3-3b-SFT-15n-8192sl-240gbsz-1.0i-0.0t-stage2-base-ST-MASKED-PLW0.5-PACKED__HF/20251104_063029_results.json", "plw0.5-packed"),

    ]

    # 2. Or tuples of (file_path, custom_display_name):
    # files = [
    #     ("/path/to/model1.json", "Baseline Model"),
    #     ("/path/to/model2.json", "Fine-tuned Model"),
    #     ("/path/to/model3.json", "Custom Architecture"),
    # ]

    plot_multiple_models(
        files,
        output_file=args.output_path,
        use_tex_text_renderer=args.use_latex_text_renderer,
        title=args.title,
        sort_files=not args.no_sort,
        output_format=args.format,
        benchmark_config=benchmark_config
    )
