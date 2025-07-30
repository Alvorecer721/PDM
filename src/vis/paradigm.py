import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verbatim_eval.controlled_expr import Results


def plot_paradigm_comparison(
    scratch_results_path: str,
    continue_results_path: str,
    metric: str = "Rouge-L",
    prefix: int = 500,
    suffix: int = 500,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Compare train-from-scratch vs continuous pretraining paradigms.
    
    Parameters
    ----------
    scratch_results_path : str
        Path to the train-from-scratch results pickle file
    continue_results_path : str
        Path to the continuous pretraining results pickle file
    metric : str
        Metric to compare (default: "Rouge-L")
    prefix : int
        Prefix length (default: 500)
    suffix : int
        Suffix length (default: 500)
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    # Load results
    scratch_results = Results.load(scratch_results_path)
    continue_results = Results.load(continue_results_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Paradigm Comparison: Train-from-Scratch vs Continuous Pretraining\n{metric} (Prefix={prefix}, Suffix={suffix})', 
                 fontsize=16)
    
    # 1. Same offset, different repetitions (top-left)
    ax = axes[0, 0]
    plot_offset_comparison(scratch_results, continue_results, metric, prefix, suffix, ax)
    
    # 2. Same repetition, different offsets (top-right)
    ax = axes[0, 1]
    plot_repetition_comparison(scratch_results, continue_results, metric, prefix, suffix, ax)
    
    # 3. Heatmap comparison (bottom row)
    plot_heatmap_comparison(scratch_results, continue_results, metric, prefix, suffix, 
                           axes[1, 0], axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_offset_comparison(scratch_results, continue_results, metric, prefix, suffix, ax):
    """Plot same offset, different repetitions comparison."""
    offsets_to_plot = [0, 32, 128, 512, 2048]  # Select a few key offsets
    
    # Colors for different models
    colors = {'scratch': 'blue', 'continue': 'red'}
    
    for i, offset in enumerate(offsets_to_plot):
        if offset not in scratch_results.offsets:
            continue
            
        scratch_means = []
        continue_means = []
        repetitions = []
        
        for rep in scratch_results.repetitions:
            try:
                scratch_stats = scratch_results.get_stats(
                    scratch_results.expr[0], rep, offset, prefix, suffix, metric
                )
                continue_stats = continue_results.get_stats(
                    continue_results.expr[0], rep, offset, prefix, suffix, metric
                )
                
                scratch_means.append(scratch_stats.mean)
                continue_means.append(continue_stats.mean)
                repetitions.append(rep)
            except:
                continue
        
        # Plot lines
        alpha = 0.7 - i * 0.1  # Fade lines for larger offsets
        ax.plot(repetitions, scratch_means, 
                color=colors['scratch'], alpha=alpha, 
                marker='o', markersize=4,
                label=f'Scratch (offset={offset})' if i == 0 else f'offset={offset}')
        ax.plot(repetitions, continue_means, 
                color=colors['continue'], alpha=alpha,
                marker='s', markersize=4,
                label=f'Continue (offset={offset})' if i == 0 else '')
    
    ax.set_xlabel('Repetitions')
    ax.set_ylabel(f'{metric} Score')
    ax.set_title('Same Offset, Different Repetitions')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')


def plot_repetition_comparison(scratch_results, continue_results, metric, prefix, suffix, ax):
    """Plot same repetition, different offsets comparison."""
    reps_to_plot = [1, 4, 16, 64, 256]  # Select key repetitions
    
    # Markers for different repetitions
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, rep in enumerate(reps_to_plot):
        if rep not in scratch_results.repetitions:
            continue
            
        scratch_means = []
        continue_means = []
        offsets = []
        
        for offset in scratch_results.offsets:
            try:
                scratch_stats = scratch_results.get_stats(
                    scratch_results.expr[0], rep, offset, prefix, suffix, metric
                )
                continue_stats = continue_results.get_stats(
                    continue_results.expr[0], rep, offset, prefix, suffix, metric
                )
                
                scratch_means.append(scratch_stats.mean)
                continue_means.append(continue_stats.mean)
                offsets.append(offset)
            except:
                continue
        
        # Plot lines
        ax.plot(offsets, scratch_means, 
                color='blue', alpha=0.7,
                marker=markers[i % len(markers)], markersize=6,
                label=f'Scratch (rep={rep})' if i == 0 else f'rep={rep}')
        ax.plot(offsets, continue_means, 
                color='red', alpha=0.7,
                marker=markers[i % len(markers)], markersize=6,
                label=f'Continue (rep={rep})' if i == 0 else '')
    
    ax.set_xlabel('Offset')
    ax.set_ylabel(f'{metric} Score')
    ax.set_title('Same Repetition, Different Offsets')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlim(left=0.8)  # Start from just before 1


def plot_heatmap_comparison(scratch_results, continue_results, metric, prefix, suffix, ax1, ax2):
    """Create heatmaps showing the difference between paradigms."""
    # Prepare data matrices
    repetitions = scratch_results.repetitions
    offsets = scratch_results.offsets
    
    scratch_matrix = np.zeros((len(repetitions), len(offsets)))
    continue_matrix = np.zeros((len(repetitions), len(offsets)))
    
    for i, rep in enumerate(repetitions):
        for j, offset in enumerate(offsets):
            try:
                scratch_stats = scratch_results.get_stats(
                    scratch_results.expr[0], rep, offset, prefix, suffix, metric
                )
                continue_stats = continue_results.get_stats(
                    continue_results.expr[0], rep, offset, prefix, suffix, metric
                )
                
                scratch_matrix[i, j] = scratch_stats.mean
                continue_matrix[i, j] = continue_stats.mean
            except:
                scratch_matrix[i, j] = np.nan
                continue_matrix[i, j] = np.nan
    
    # Calculate difference (continue - scratch)
    diff_matrix = continue_matrix - scratch_matrix
    
    # Plot absolute difference heatmap
    im1 = ax1.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax1.set_xticks(range(len(offsets)))
    ax1.set_xticklabels(offsets, rotation=45)
    ax1.set_yticks(range(len(repetitions)))
    ax1.set_yticklabels(repetitions)
    ax1.set_xlabel('Offset')
    ax1.set_ylabel('Repetitions')
    ax1.set_title(f'Difference: Continue - Scratch\n(Red = Continue better, Blue = Scratch better)')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(f'Δ{metric}')
    
    # Plot relative difference heatmap (percentage improvement)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff = (diff_matrix / scratch_matrix) * 100
    
    im2 = ax2.imshow(relative_diff, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_xticks(range(len(offsets)))
    ax2.set_xticklabels(offsets, rotation=45)
    ax2.set_yticks(range(len(repetitions)))
    ax2.set_yticklabels(repetitions)
    ax2.set_xlabel('Offset')
    ax2.set_ylabel('Repetitions')
    ax2.set_title('Relative Improvement (%)\n(Continue vs Scratch)')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('% Improvement')


def create_summary_table(
    scratch_results_path: str,
    continue_results_path: str,
    metric: str = "Rouge-L",
    prefix: int = 500,
    suffix: int = 500,
    selected_offsets: Optional[List[int]] = None,
    selected_reps: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create a summary table comparing the two paradigms.
    
    Returns a DataFrame with statistics for easy analysis.
    """
    scratch_results = Results.load(scratch_results_path)
    continue_results = Results.load(continue_results_path)
    
    if selected_offsets is None:
        selected_offsets = [0, 32, 128, 512, 2048]
    if selected_reps is None:
        selected_reps = [1, 4, 16, 64, 256]
    
    data = []
    
    for offset in selected_offsets:
        if offset not in scratch_results.offsets:
            continue
            
        for rep in selected_reps:
            if rep not in scratch_results.repetitions:
                continue
                
            try:
                scratch_stats = scratch_results.get_stats(
                    scratch_results.expr[0], rep, offset, prefix, suffix, metric
                )
                continue_stats = continue_results.get_stats(
                    continue_results.expr[0], rep, offset, prefix, suffix, metric
                )
                
                improvement = continue_stats.mean - scratch_stats.mean
                relative_improvement = (improvement / scratch_stats.mean) * 100
                
                data.append({
                    'Offset': offset,
                    'Repetitions': rep,
                    'Scratch_Mean': scratch_stats.mean,
                    'Continue_Mean': continue_stats.mean,
                    'Absolute_Improvement': improvement,
                    'Relative_Improvement_%': relative_improvement,
                    'Scratch_Std': scratch_stats.std,
                    'Continue_Std': continue_stats.std
                })
            except:
                continue
    
    df = pd.DataFrame(data)
    return df


def print_paradigm_comparison_table(
    scratch_results_path: str,
    continue_results_path: str,
    metric: str = "Rouge-L",
    prefix: int = 500,
    suffix: int = 500,
    selected_offsets: Optional[List[int]] = None,
    selected_reps: Optional[List[int]] = None,
    show_std: bool = True
):
    """
    Print a formatted comparison table between train-from-scratch and continuous pretraining.
    """
    scratch_results = Results.load(scratch_results_path)
    continue_results = Results.load(continue_results_path)
    
    if selected_offsets is None:
        selected_offsets = [0, 32, 128, 512, 2048]
    if selected_reps is None:
        selected_reps = [1, 4, 16, 64, 256]
    
    print(f"\n=== Paradigm Comparison: {metric} (Prefix={prefix}, Suffix={suffix}) ===")
    print()
    
    # Build header
    if show_std:
        value_width = 15  # "  0.965 ± 0.135"
        model_col_width = 15
    else:
        value_width = 9   # "  0.965"
        model_col_width = 9
    
    # First header row - models
    header1_parts = [f"{'':>16} |"]
    for offset in selected_offsets:
        # Each offset group contains both models
        group_width = 2 * model_col_width + 3  # 2 models + separator
        header1_parts.append(f" Offset={offset:^{group_width}} |")
    
    # Second header row - repetitions and model names
    header2_parts = [f"{'Repetitions':>16} |"]
    for offset in selected_offsets:
        header2_parts.append(f" {'Scratch':>{model_col_width}} | {'Continue':>{model_col_width}} |")
    
    header1 = "".join(header1_parts)
    header2 = "".join(header2_parts)
    
    print(header1)
    print(header2)
    print("-" * len(header2))
    
    # Print data rows
    for rep in selected_reps:
        if rep not in scratch_results.repetitions:
            continue
            
        row_parts = [f"{rep:16} |"]
        
        for offset in selected_offsets:
            if offset not in scratch_results.offsets:
                if show_std:
                    row_parts.append(f" {'N/A':^{model_col_width}} | {'N/A':^{model_col_width}} |")
                else:
                    row_parts.append(f" {'N/A':^{model_col_width}} | {'N/A':^{model_col_width}} |")
                continue
            
            try:
                scratch_stats = scratch_results.get_stats(
                    scratch_results.expr[0], rep, offset, prefix, suffix, metric
                )
                continue_stats = continue_results.get_stats(
                    continue_results.expr[0], rep, offset, prefix, suffix, metric
                )
                
                if show_std:
                    scratch_str = f"{scratch_stats.mean:.3f} ± {scratch_stats.std:.3f}"
                    continue_str = f"{continue_stats.mean:.3f} ± {continue_stats.std:.3f}"
                else:
                    scratch_str = f"{scratch_stats.mean:.3f}"
                    continue_str = f"{continue_stats.mean:.3f}"
                
                row_parts.append(f" {scratch_str:>{model_col_width}} | {continue_str:>{model_col_width}} |")
            except:
                row_parts.append(f" {'Error':^{model_col_width}} | {'Error':^{model_col_width}} |")
        
        print("".join(row_parts))
    
    # Print improvement summary
    print("\n=== Improvement Summary (Continue vs Scratch) ===")
    print()
    print(f"{'Offset':>8} | {'Avg Improvement':>15} | {'Max Improvement':>15} | {'Min Improvement':>15}")
    print("-" * 60)
    
    for offset in selected_offsets:
        if offset not in scratch_results.offsets:
            continue
            
        improvements = []
        for rep in selected_reps:
            if rep not in scratch_results.repetitions:
                continue
            try:
                scratch_stats = scratch_results.get_stats(
                    scratch_results.expr[0], rep, offset, prefix, suffix, metric
                )
                continue_stats = continue_results.get_stats(
                    continue_results.expr[0], rep, offset, prefix, suffix, metric
                )
                improvements.append(continue_stats.mean - scratch_stats.mean)
            except:
                continue
        
        if improvements:
            avg_imp = np.mean(improvements)
            max_imp = np.max(improvements)
            min_imp = np.min(improvements)
            print(f"{offset:8} | {avg_imp:15.4f} | {max_imp:15.4f} | {min_imp:15.4f}")


if __name__ == "__main__":
    # Example usage
    scratch_path = '/iopsstor/scratch/cscs/xyixuan/PDM/results/sparse/llama3-1b-15n-8192sl-60gbsz-standard/offset_0_1_2_4_8_16_32_64_128_256_512_1024_2048_prefix_500_suffix_500_greedy.pkl'
    continue_path = '/iopsstor/scratch/cscs/xyixuan/PDM/results/sparse/llama3-1b-15n-8192sl-60gbsz-continue/offset_0_1_2_4_8_16_32_64_128_256_512_1024_2048_prefix_500_suffix_500_greedy.pkl'
    
    # Print comparison table
    print_paradigm_comparison_table(
        scratch_path,
        continue_path,
        metric='Rouge-L',
        prefix=500,
        suffix=500
    )
    
    # Create comparison plot
    plot_paradigm_comparison(
        scratch_path,
        continue_path,
        metric='Rouge-L',
        prefix=500,
        suffix=500,
        save_path='/iopsstor/scratch/cscs/xyixuan/PDM/results/vis/paradigm_comparison.png'
    )
    
    # Create summary table
    df = create_summary_table(scratch_path, continue_path)
    print("\nDetailed Summary Table:")
    print(df.to_string(index=False))
    
    # Save table
    df.to_csv('/iopsstor/scratch/cscs/xyixuan/PDM/results/vis/paradigm_comparison_summary.csv', index=False)