"""
plot_eval_results.py:

generate bar-plot comparing different models on different benchmarks.
Specify the path to respective json_files (results of PDM eval) down below and run the script like this:
    python -m lm_eval.plot_eval_results --output-path=[PATH_TO_RES_FILE] [--use-latex-text-renderer] [--title="Custom Title"] [--no-sort] [--format=png]

You can specify files in two ways:
    1. Just file paths: The model name will be extracted automatically from the path
    2. Tuples of (file_path, display_name): Use custom display names in the legend

By default, files are sorted by their ratio pattern (e.g., 0.6i-0.4t). Use --no-sort to preserve the order as specified.
Output format defaults to PDF. Use --format to specify png, jpg, svg, or other matplotlib-supported formats.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import re

RATIO_PATTERN = re.compile(r'(\d\.\d{1,2})(?:i)?-(\d\.\d{1,2})(?:t)?')

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
            display_name = f"Pre-trained with {x}\\% image and {y}\\% text tokens"
        else:
            display_name = f"{x}-{y}"
    else:
        display_name = parent  # fallback if no match

    return path, display_name

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


def plot_multiple_models(json_files: list, output_file: str, use_tex_text_renderer: bool = False, title: str = None, sort_files: bool = True, output_format: str = 'pdf'):
    """
    Plot benchmark results for multiple models.

    Args:
        json_files: List of file paths (str) or tuples of (file_path, display_name)
        output_file: Path to save the output plot
        use_tex_text_renderer: Whether to use LaTeX for text rendering
        title: Custom title for the plot (default: None, no title shown)
        sort_files: Whether to sort files by their ratio pattern (default: True)
        output_format: Output file format (default: 'pdf'). Supports 'png', 'jpg', 'svg', etc.
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
        labels, accs, errs = load_benchmarks(jf)
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
    plt.xticks(x, all_labels, rotation=45, ha="right", fontsize=14)
    plt.ylabel(r"Accuracy", weight="bold", fontsize=16)

    # Add title if provided
    if title:
        plt.title(title, fontsize=18, weight='bold', pad=20)

    # Place the legend inside the plot (bottom right) with semi-transparent background
    # Use LaTeX formatting for legend title only if LaTeX renderer is active
    legend_title = r"\textbf{Models}" if use_tex_text_renderer else "Models"
    plt.legend(
        title=legend_title,
        loc='lower right',  # Place the legend inside the plot at the bottom right
        fontsize=10,
        title_fontsize=11,
        frameon=True,  # Keep the frame for the legend
        framealpha=0.7,  # Set transparency (0.0 is fully transparent, 1.0 is fully opaque)
        ncol=1,  # One column, but broad
        borderpad=1,  # Add padding around the legend box
        labelspacing=1.5,  # Increase space between legend entries
        handlelength=3,  # Increase length of the legend box handles
        columnspacing=1.5  # Add more space between legend columns (if you had more columns)
    )

    # Add gridlines and increase the prominence of the axes
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Light grid lines for better readability

    # Ensure the layout is tight and the plot is clean
    plt.tight_layout()

    # Save the figure in the specified format
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format=output_format)
    plt.close()

    print(f"✅ Plot saved to {output_file}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--use-latex-text-renderer", action="store_true", help="Use LaTeX to render text in plots. Only works with local tex installation")
    argparser.add_argument("--output-path", type=str, default="/iopsstor/scratch/cscs/nirmiger/PDM/results/lm_eval/plots/llm_comparison.pdf", help="Path to save the plot")
    argparser.add_argument("--title", type=str, default=None, help="Custom title for the plot (optional)")
    argparser.add_argument("--no-sort", action="store_true", help="Disable automatic sorting of files by ratio pattern (preserves input order)")
    argparser.add_argument("--format", type=str, default="pdf", help="Output file format (default: pdf). Supports png, jpg, svg, etc.")
    args = argparser.parse_args()

    # Specify the paths to the JSON files that contain the benchmark results
    # You can use either:
    # 1. Plain file paths (names will be auto-extracted):
    files = [
        "/Users/nicolairmiger/PDM/results/lm_eval/Llama-3.2-3B/__iopsstor__scratch__cscs__nirmiger__Llama-3.2-3B/results_2025-09-02T14-11-42.014991.json",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.6i-0.4t/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-15n-8192sl-120gbsz-0.6i-0.4t__HF/results_2025-09-15T09-57-15.632974.json",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.8i-0.2t/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-15n-8192sl-120gbsz-0.8i-0.2t__HF/results_2025-09-15T10-13-43.507556.json",
        "/Users/nicolairmiger/PDM/results/lm_eval/llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-27000/__iopsstor__scratch__cscs__nirmiger__Megatron-LM__logs__Meg-Runs__image-extension__llama3-3b-15n-8192sl-120gbsz-0.9i-0.1t-27000__HF/results_2025-09-15T09-59-25.371298.json",
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
        output_format=args.format
    )
