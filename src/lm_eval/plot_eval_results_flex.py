"""
plot_eval_results_flex.py:

Flexible plotting utility for lm-eval results. Automatically collects all
result files from specified folders.

Usage:
    python -m lm_eval.plot_eval_results_flex \
        --output-path output.pdf \
        [--benchmarks benchmark1,benchmark2] \
        [--use-latex-text-renderer] \
        [--title "Custom Title"] \
        [--format png] \
        [--instruct-bench] \
        [--multi-modal-bench]

Define models in the script at the bottom:
    models = [
        {"display_name": "Model A", "path_to_result_files": "/path/to/results/model_a"},
        {"display_name": "Model B", "path_to_result_files": "/path/to/results/model_b"},
    ]

Features:
    - Recursively searches each folder for results_*.json files
    - Automatically collects and merges results from multiple JSON files
    - Validates that all models have the same benchmark set
    - Raises errors on duplicate benchmark keys within a single model
    - Supports filtering benchmarks via --benchmarks flag
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Benchmark configurations: (metric_key, stderr_key)
DEFAULT_BENCHMARKS = {
    "arc_challenge": ("acc,none", "acc_stderr,none"),
    "arc_easy": ("acc,none", "acc_stderr,none"),
    "hellaswag": ("acc,none", "acc_stderr,none"),
    "piqa": ("acc,none", "acc_stderr,none"),
    "winogrande": ("acc,none", "acc_stderr,none"),
    "mmlu": ("acc,none", "acc_stderr,none"),
    "commonsense_qa":  ("acc,none", "acc_stderr,none"),
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
    "ai2d": ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
}


def collect_result_files(folder_path: str) -> list[Path]:
    """
    Recursively find all results_*.json files in folder.

    Args:
        folder_path: Path to the folder to search

    Returns:
        List of Path objects for matching files

    Raises:
        ValueError: If folder does not exist
    """
    path = Path(folder_path)
    if not path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    return list(path.rglob("results_*.json"))


def load_results_from_folder(
    folder_path: str, benchmark_config: dict
) -> tuple[list[str], list[float], list[float]]:
    """
    Load all results from a folder by recursively collecting JSON files.

    Args:
        folder_path: Path to search for result files
        benchmark_config: Dict mapping benchmark names to (metric_key, stderr_key)

    Returns:
        tuple: (labels, accuracies, errors) lists

    Raises:
        ValueError: If no result files found or duplicate benchmark keys found across files
    """
    files = collect_result_files(folder_path)
    if not files:
        raise ValueError(f"No results_*.json files found in {folder_path}")

    # Collect all results, checking for duplicates
    all_results = {}
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
        results = data.get("results", {})
        for key in results:
            if key in all_results:
                raise ValueError(
                    f"Duplicate benchmark '{key}' found in {folder_path}. "
                    f"Files contain overlapping results."
                )
            all_results[key] = results[key]

    # Extract metrics using benchmark config
    labels, accuracies, errors = [], [], []
    for name, (metric_key, stderr_key) in benchmark_config.items():
        if name not in all_results:
            continue
        vals = all_results[name]
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


def validate_consistent_benchmarks(
    all_labels: list[list[str]], model_names: list[str]
) -> None:
    """
    Ensure all models have the same benchmark labels.

    Args:
        all_labels: List of label lists, one per model
        model_names: List of model display names

    Raises:
        ValueError: If models have inconsistent benchmark sets
    """
    if not all_labels:
        return

    reference = set(all_labels[0])
    reference_name = model_names[0]

    for i, labels in enumerate(all_labels[1:], 1):
        current = set(labels)
        if current != reference:
            missing = reference - current
            extra = current - reference
            msg = f"Model '{model_names[i]}' has inconsistent benchmarks compared to '{reference_name}'.\n"
            if missing:
                msg += f"  Missing: {missing}\n"
            if extra:
                msg += f"  Extra: {extra}"
            raise ValueError(msg)


def plot_models(
    models: list[dict],
    output_file: str,
    benchmark_config: dict,
    use_tex_text_renderer: bool = False,
    title: str = None,
    output_format: str = "pdf",
):
    """
    Main plotting function that creates a bar plot comparing models.

    Args:
        models: List of dicts with 'display_name' and 'path_to_result_files' keys
        output_file: Path to save the output plot
        benchmark_config: Dict mapping benchmark names to (metric_key, stderr_key)
        use_tex_text_renderer: Whether to use LaTeX for text rendering
        title: Custom title for the plot (optional)
        output_format: Output file format (default: 'pdf')
    """
    # Setup matplotlib/seaborn styling
    if use_tex_text_renderer:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

    sns.set_theme(style="whitegrid", font="serif", font_scale=1.5)

    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.figsize": (12, 6),
            "lines.linewidth": 2,
            "axes.linewidth": 1.0,
            "legend.title_fontsize": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "font.family": "serif",
        }
    )

    # Load results for each model
    all_labels = []
    all_accuracies = []
    all_errors = []
    model_names = []

    for model in models:
        display_name = model["display_name"]
        folder_path = model["path_to_result_files"]

        labels, accs, errs = load_results_from_folder(folder_path, benchmark_config)

        if not labels:
            print(f"Warning: No matching benchmarks found for model '{display_name}' in {folder_path}")
            continue

        all_labels.append(labels)
        all_accuracies.append(accs)
        all_errors.append(errs)
        model_names.append(display_name)

    if not model_names:
        raise ValueError("No valid models with matching benchmarks found.")

    # Validate consistency across models
    validate_consistent_benchmarks(all_labels, model_names)

    # Create the x locations for each benchmark
    x = np.arange(len(all_labels[0]))
    num_models = len(model_names)
    width = 0.8 / num_models

    # Create figure
    plt.figure(figsize=(12, 6))

    # Use colorblind-friendly colors
    colors = sns.color_palette("colorblind", num_models)

    # Plot each model's bars with error bars
    for i, (accs, errs) in enumerate(zip(all_accuracies, all_errors)):
        offset = (i - num_models / 2) * width + width / 2
        plt.bar(
            x + offset,
            accs,
            width,
            yerr=errs,
            capsize=4,
            label=model_names[i],
            color=colors[i],
        )

    # Adjust axis labels and ticks
    plt.xticks(x, all_labels[0], rotation=0, ha="center", fontsize=14)
    plt.ylabel(r"Accuracy", weight="bold", fontsize=16)

    # Add title if provided
    if title:
        plt.title(title, fontsize=18, weight="bold", pad=20)

    # Add gridlines
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Place the legend below the plot
    legend_title = r"\textbf{Models}" if use_tex_text_renderer else "Models"
    ncol = min(num_models, 4)

    plt.legend(
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
        ncol=ncol,
        borderpad=1,
        labelspacing=0.5,
        handlelength=2,
        columnspacing=1.5,
        fancybox=True,
        shadow=True,
    )

    # Ensure the layout is tight
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25 + (0.03 * ((num_models - 1) // ncol)))

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight", format=output_format)
    plt.close()

    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flexible lm-eval results plotting utility"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to include (filters benchmark_config)",
    )
    parser.add_argument(
        "--use-latex-text-renderer",
        action="store_true",
        help="Use LaTeX to render text in plots",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        help="Output file format (default: pdf). Supports png, jpg, svg, etc.",
    )
    parser.add_argument(
        "--instruct-bench",
        action="store_true",
        help="Use instruct benchmarks instead of default benchmarks",
    )
    parser.add_argument(
        "--multi-modal-bench",
        action="store_true",
        help="Use multi-modal benchmarks instead of default benchmarks",
    )
    args = parser.parse_args()

    # Build benchmark config based on flags
    benchmark_config = (
        DEFAULT_BENCHMARKS if not (args.instruct_bench or args.multi_modal_bench) else {}
    )

    if args.multi_modal_bench:
        benchmark_config = {**benchmark_config, **MULTIMODAL_BENCHMARKS}

    if args.instruct_bench:
        benchmark_config = {**benchmark_config, **INSTRUCT_BENCHMARKS}

    # Filter benchmarks if specified
    if args.benchmarks:
        filter_list = [b.strip() for b in args.benchmarks.split(",")]
        benchmark_config = {
            k: v for k, v in benchmark_config.items() if k in filter_list
        }
        if not benchmark_config:
            raise ValueError(
                f"No benchmarks matched filter: {filter_list}. "
                f"Available benchmarks depend on --instruct-bench/--multi-modal-bench flags."
            )

    # =========================================================================
    # Define models here (edit this section)
    # =========================================================================
    # Each model is a dict with:
    #   - "display_name": Name to show in the legend
    #   - "path_to_result_files": Folder path to recursively search for results_*.json
    #
    # Example:
    models = [
        {
            "display_name": "Apertus8B-base",
            "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/extended_model2",
        },
        {
            "display_name": "Apertus8B-s1-cd",
            "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/apertus-8b-img-pretrain-64nodes-gbs1024-mbs1-steps7003-img0.9-text0.1-seqlen8192_cooldown",
        },
        # {
        #     "display_name": "Apertus8B-s1-nogf-cd",
        #     "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/apertus-8b-img-pretrain-64nodes-gbs1024-mbs1-steps7003-img0.9-text0.1-seqlen8192-NOGOLDFISH_cooldown",
        # },
        {
            "display_name": "Apertus8B-s1-nogf-nocd",
            "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/apertus-8b-img-pretrain-64nodes-gbs1024-mbs1-steps7003-img0.9-text0.1-seqlen8192-NOGOLDFISH_no_cooldown",
        },
        {
            "display_name": "Apertus8B-s1-selGf-Ocr-nocd",
            "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/apertus-8b-img-pretrain-64nodes-gbsz1024-mbs1-step7878-i0.9-t0.1-seqlen8192-selectiveGf-ocr-longwarmup",
        },
        # {
        #     "display_name": "Apertus8B-s2-nogf-cd",
        #     "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/apertus-8b-img-pretrain-PAIRED-64nodes-gbs1024-mbs1-steps10072-img0.9-text0.1-seqlen8192-NOGOLDFISH_cooldown",
        # },
        # {
        #     "display_name": "Apertus8B-s2-nogf-nocd",
        #     "path_to_result_files": "/users/rkreft/PDM/results/lm_eval/apertus-8b-img-pretrain-PAIRED-64nodes-gbs1024-mbs1-steps10072-img0.9-text0.1-seqlen8192-NOGOLDFISH_no_cooldown",
        # },
    ]
    # =========================================================================

    plot_models(
        models,
        args.output_path,
        benchmark_config,
        use_tex_text_renderer=args.use_latex_text_renderer,
        title=args.title,
        output_format=args.format,
    )
