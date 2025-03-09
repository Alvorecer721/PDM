import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

def plot_nll_distributions_ridge(results_dict, model_key, upper_quantile=1.):
    """
    Create a ridge plot of NLL distributions for each subkey in results_dict[model_key].
    
    Parameters
    ----------
    results_dict : dict
        The dictionary containing all model results (e.g., goldfish_res_greedy).
    model_key : str
        The dictionary key for the specific model 
        (e.g., 'llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_1984000').
    """

    # 1. Reshape your data into a DataFrame with columns: [ 'subkey', 'nll' ]
    data_records = []
    for subkey, value_dict in results_dict[model_key].items():
        scores = np.array(value_dict['NLL']['scores'])
        for score in scores:
            data_records.append({'subkey': subkey, 'Negative Log Likelihood': score})
    
    df = pd.DataFrame(data_records)

    # (Optional) Clip outliers or large values
    lower = df["Negative Log Likelihood"].quantile(0.00)
    upper = df["Negative Log Likelihood"].quantile(upper_quantile)
    df["Trimmed Negative Log Likelihood"] = df["Negative Log Likelihood"].clip(lower, upper)

    # 2. Initialize a palette and a FacetGrid
    unique_subkeys = df["subkey"].unique()
    pal = sns.cubehelix_palette(len(unique_subkeys), rot=-.25, light=.7)
    g = sns.FacetGrid(
        df, 
        row="subkey",                     # each subkey in its own row
        hue="subkey",                     # color by subkey
        aspect=20,                        # make plots much wider than tall
        height=.5,                        # the vertical height of each plot
        palette=pal
    )

    # 3. Plot the ridgeline KDEs (filled, then outline)
    g.map(
        sns.kdeplot, 
        "Trimmed Negative Log Likelihood",
        bw_adjust=.5,
        clip_on=False,
        fill=True,
        alpha=1, 
        linewidth=1.5
    )

    g.map(
        sns.kdeplot, 
        "Trimmed Negative Log Likelihood", 
        clip_on=False, 
        color="w", 
        lw=2, 
        bw_adjust=.5
    )

    # 4. Reference line at y=0 (for each row)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # 5. Define and apply a small labeling function to place text within each subplot
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .3, label, 
                fontweight="bold", 
                color=color, 
                ha="left", 
                va="center", 
                transform=ax.transAxes)

    # Map the labeling function to one of the variables (not the trimmed one!)
    g.map(label, "Negative Log Likelihood")

    # 6. Adjust the subplot spacing so that subplots overlap
    g.figure.subplots_adjust(hspace=-.05)

    # 7. Remove or simplify unneeded axis details
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # --- ADDING TITLE AND AXIS LABELS ---
    # 8. Set a main title for the entire figure
    g.fig.suptitle(
        model_key, 
        x=0.5,              # center the title
        y=1.03,             # adjust if it overlaps the topmost subplot
        fontsize=10
    )
    
    # 9. Optionally adjust the top margin to ensure the title fits
    g.figure.subplots_adjust(
        hspace=-0.25,      # Increase overlap between subplots
        left=0.1,          # Increase left margin for labels
        right=0.9,         # Adjust right margin
        top=0.95,          # Adjust top margin for title
        bottom=0.1         # Adjust bottom margin
    )
    
    # 10. Set the x-axis label for all facets
    g.set_axis_labels("Trimmed Negative Log Likelihood", "")

    plt.show()


def plot_batch_distribution(dataset_index_path, show_n_batches, batch_size, log_y=True, reference_line=True):
    dataset_index = np.load(dataset_index_path)
    dataset_index_shown = dataset_index[:show_n_batches * batch_size]
    
    # Get overall stats
    unique_all, counts_all = np.unique(dataset_index, return_counts=True)
    expected_samples_per_batch = counts_all / len(dataset_index) * 60

    
    windows = []
    batch_numbers = []
    source_names = {
        0: 'Fineweb',
        1: 'rep 128',
        2: 'rep 256', 
        3: 'rep 512',
        4: 'rep 1024',
        5: 'rep 2048'
    }

    for expected_samples in zip(unique_all, expected_samples_per_batch):
        print(f'Expected {source_names[expected_samples[0]]} samples per batch: {expected_samples[1]:.2f}')
    
    for start in range(0, len(dataset_index_shown)-batch_size, batch_size):
        window_data = dataset_index_shown[start:start+batch_size]
        unique, count = np.unique(window_data, return_counts=True)
        windows.append(dict(zip(unique, count)))
        batch_numbers.append(start // batch_size)
        
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(18, 6))
    
    for source in sorted(np.unique(dataset_index_shown)):
        y = [window.get(source, 0) for window in windows]
        ax.plot(batch_numbers, y, label=source_names[source], linestyle='--')
        
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    ax.set_title(f'Data Loading Frequency Over First {show_n_batches} Batches (Batch Size = {batch_size})')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, color='gray')
    if log_y:
        ax.set_yscale('log')
    ax.legend(facecolor='white', edgecolor='black')
    
    tick_interval = max(1, show_n_batches // 10)
    ax.set_xticks(np.arange(0, show_n_batches, tick_interval))

    # Add expected sample lines
    if reference_line:
        for source, expected in zip(sorted(np.unique(dataset_index)), expected_samples_per_batch):
            ax.axhline(y=expected, color='red', linestyle='solid', alpha=0.5, 
                        label=f'Expected {source_names[source]}' if source == 0 else "")

    plt.tight_layout()
    plt.show()

def compare_metric_distributions(
    expr_list: list[str],            # List of experiment names to compare
    repetitions: list[int],          # List of repetition numbers
    offsets: list[int],              # List of offset values
    results_dict: dict,              # Dictionary of Results objects, keyed by experiment name
    metric: str = 'Rouge-L',         # Metric to plot
    prefix_length: int = 500,        # Prefix length to use
    suffix_length: int = 500,        # Suffix length to use
    figsize: tuple = None,           # Optional custom figure size
    custom_labels: list = None       # Custom labels for the legend
):
    """
    Plot the distribution of a specified metric for two experiments side by side.
    
    Args:
        expr_list: List of experiment names to compare
        repetitions: List of repetition numbers to include
        offsets: List of offset values to include
        results_dict: Dictionary of Results objects, keyed by experiment name
        metric: Metric to plot (default: 'Rouge-L')
        prefix_length: Prefix length to use (default: 500)
        suffix_length: Suffix length to use (default: 500)
        figsize: Optional custom figure size (default: calculated based on subplots)
        custom_labels: Custom labels for the legend (default: None, will use H values)
    """
    # Validate inputs
    if len(expr_list) != 2:
        raise ValueError("This function is designed to compare exactly two experiments")
    
    # Find dimensions
    n_reps = len(repetitions)
    n_offsets = len(offsets)
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (6 * n_reps, 5 * n_offsets)
    
    # Set the style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create figure and subplots: rows for offsets, columns for repetitions
    fig, axes = plt.subplots(n_offsets, n_reps, figsize=figsize)
    
    # Make axes 2D if it's not already
    if n_offsets == 1 and n_reps == 1:
        axes = np.array([[axes]])
    elif n_offsets == 1:
        axes = axes.reshape(1, -1)
    elif n_reps == 1:
        axes = axes.reshape(-1, 1)
    
    # Define more visually distinct colors
    colors = ['#1f77b4', '#d62728']  # Blue and red with better contrast
    
    # Extract H values from model names for legend if custom labels not provided
    if custom_labels is None:
        custom_labels = []
        for expr in expr_list:
            # Try to extract H value from model name
            if 'h-' in expr:
                h_value = expr.split('h-')[1].split('-')[0].split('sl')[0]
                custom_labels.append(f"H={h_value}")
            else:
                custom_labels.append(expr)
    
    # Plot each offset and repetition combination
    for offset_idx, offset in enumerate(offsets):
        for rep_idx, rep in enumerate(repetitions):
            ax = axes[offset_idx, rep_idx]
            
            all_scores = []
            # First pass to collect all scores for bin calculation
            for expr_idx, expr in enumerate(expr_list):
                results_obj = results_dict[expr]
                metric_data = results_obj.get_stats(
                    expr, rep, offset, prefix_length, suffix_length, metric
                )
                all_scores.extend(metric_data.scores)
            
            # Calculate common bins for both histograms
            if len(set(all_scores)) <= 1:  # All values are the same
                bins = 1
            else:
                bins = np.linspace(min(all_scores), max(all_scores), 21)  # 20 bins
            
            # Second pass to plot with common bins
            for expr_idx, expr in enumerate(expr_list):
                results_obj = results_dict[expr]
                metric_data = results_obj.get_stats(
                    expr, rep, offset, prefix_length, suffix_length, metric
                )
                scores = metric_data.scores
                
                # Plot with more transparency and hatching for better distinction
                alpha = 0.7
                hatch = None if expr_idx == 0 else '//'
                
                ax.hist(scores, bins=bins, color=colors[expr_idx], alpha=alpha,
                       edgecolor='black', linewidth=1.0, label=custom_labels[expr_idx],
                       hatch=hatch)
            
            # Move legend outside plot
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), 
                     fontsize=12, frameon=True, framealpha=1, edgecolor='black')
            
            # Set labels with larger font
            ax.set_xlabel(f'{metric} Score', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            
            # For Rouge-L, limit to [0, 1] range
            if metric == 'Rouge-L':
                ax.set_xlim(0, 1)
            
            # Set grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Only show y-label for first column
            if rep_idx > 0:
                ax.set_ylabel('')
            
            # Only show x-label for last row
            if offset_idx < n_offsets - 1:
                ax.set_xlabel('')
    
    # Add row labels for offsets
    # for offset_idx, offset in enumerate(offsets):
    #     axes[offset_idx, 0].text(-0.3, 0.5, f'Offset {offset}', fontsize=16,
    #                             rotation=90, transform=axes[offset_idx, 0].transAxes,
    #                             verticalalignment='center')
    
    # Add column labels for repetitions
    # for rep_idx, rep in enumerate(repetitions):
    #     axes[0, rep_idx].text(0.5, 1.15, f'Rep {rep}', fontsize=16,
    #                          transform=axes[0, rep_idx].transAxes,
    #                          horizontalalignment='center')
    
    # First call tight_layout with padding
    # plt.tight_layout(rect=[0, 0.05, 0.95, 0.95])  # Leave space for legend and titles
    
    return fig, axes


def plot_repetition_metric_dists(expr: str, repetitions: list[int], offsets: list[int], dict_results: dict[str, dict[str, list[float]]], metric: str):
    """Plot the distribution of exact match LCS lengths for all repetitions."""
    # Find all repetitions and offsets for this expression
    n_reps = len(repetitions)
    n_offsets = len(offsets)  # Number of offsets
    
    # Create figure and subplots: rows for offsets, columns for repetitions
    fig, axes = plt.subplots(n_offsets, n_reps, figsize=(5*n_reps, 4*n_offsets))
    
    # Create a color map for different offsets
    colors = plt.cm.tab20(np.linspace(0, 1, n_offsets))
    
    # Make axes 2D if it's not already
    if n_offsets == 1 and n_reps == 1:
        axes = np.array([[axes]])
    elif n_offsets == 1:
        axes = axes.reshape(1, -1)
    elif n_reps == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each offset and repetition combination
    for offset_idx, offset in enumerate(offsets):
        for rep_idx, rep in enumerate(repetitions):
            ax = axes[offset_idx, rep_idx]
            
            if metric == 'TTR':
                score = dict_results[expr][rep][offset]['TTR_gen']['scores']
                ref = dict_results[expr][rep][offset]['TTR_ref']['scores']
                # Plot both distributions
                ax.hist(score, bins=20, color=colors[offset_idx], edgecolor='black', 
                       linewidth=1.2, alpha=0.7, label='Generated')
                ax.hist(ref, bins=20, color='gray', edgecolor='black', 
                       linewidth=1.2, alpha=0.5, label='Reference')
                ax.legend(fontsize=12)
            else:
                score = dict_results[expr][rep][offset][metric]['scores']
                ax.hist(score, bins=20, color=colors[offset_idx], edgecolor='black', linewidth=1.2)
            
            ax.set_xlabel(f'{metric} Score')
            ax.set_ylabel('Frequency')
            if metric == 'Rouge-L':
                ax.set_xlim(0, 1)
            
            # Only show y-label for first column
            if rep_idx > 0:
                ax.set_ylabel('')
            
            # Only show x-label for last row
            if offset_idx < n_offsets - 1:
                ax.set_xlabel('')
    
    # Add row labels for offsets
    for offset_idx, offset in enumerate(offsets):
        axes[offset_idx, 0].text(-0.3, 0.5, f'Offset {offset}', fontsize=20,
                                rotation=90, transform=axes[offset_idx, 0].transAxes,
                                verticalalignment='center')
    
    # Add column labels for repetitions
    for rep_idx, rep in enumerate(repetitions):
        axes[0, rep_idx].text(0.5, 1.2, f'Rep {rep}', fontsize=20,
                             transform=axes[0, rep_idx].transAxes,
                             horizontalalignment='center')
    
    # First call tight_layout with padding
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for title

    # Then add title at the bottom center
    fig.text(0.5, 0.01, f'{metric} Score for {expr}',
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=30)