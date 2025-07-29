import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Union, Optional

def plot_dense_gutenberg_offset_power_law(results_dict, model_name, selected_repetitions=[1, 2, 4, 8, 16, 32, 64], log_y=True, show_fit_curve=True):
    """
    Create a single log-log plot showing the relationship between offset and Rouge-L scores
    for selected repetition counts with improved legend placement and styling.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results objects
    model_name : str
        Name of the model to plot
    selected_repetitions : list
        List of repetition values to include in the plot
    log_y : bool
        Whether to use logarithmic scale for y-axis (default: True)
    show_fit_curve : bool
        Whether to display the power law fit curves (default: True)
    """
    plt.figure(figsize=(12, 6))
    
    result_obj = results_dict[model_name]
    expr_name = result_obj.expr[0]
    prefix = result_obj.prefixes[0]
    suffix = result_obj.suffixes[0]
    
    # Get all offsets
    offsets = sorted(result_obj.offsets)
    
    # Create color map for repetition values
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_repetitions)))
    
    # For each selected repetition, plot offset vs. Rouge-L
    for i, rep in enumerate(selected_repetitions):
        if rep not in result_obj.repetitions:
            continue
            
        # Collect data for this repetition
        valid_offsets = []
        rouge_scores = []
        
        for offset in offsets:
            try:
                stats = result_obj.get_stats(expr_name, rep, offset, prefix, suffix, 'Rouge-L')
                valid_offsets.append(offset if offset > 0 else 0.5)  # Avoid log(0)
                rouge_scores.append(stats.mean)
            except KeyError:
                continue
        
        # Plot the data points
        plt.plot(valid_offsets, rouge_scores, 'o-', 
                color=colors[i], 
                markersize=8)
        
        # Fit a power law if we have enough points
        if len(valid_offsets) > 2:
            # Convert to log space for linear fit
            log_x_values = np.log(np.array(valid_offsets))
            log_y_values = np.log(np.array(rouge_scores))
            
            # Linear fit in log space
            coef = np.polyfit(log_x_values, log_y_values, 1)
            
            # Extract power law parameters
            a = np.exp(coef[1])
            b = coef[0]
            
            # Plot fitted power law if requested
            if show_fit_curve:
                x_fit = np.array([min(valid_offsets), max(valid_offsets)])
                y_fit = a * x_fit**b
                plt.plot(x_fit, y_fit, '--', color=colors[i], alpha=0.7)
                
                # Use dashed line in legend to match fit curve
                line_style = '--'
            else:
                # Use solid line in legend to match data line
                line_style = '-'
            
            # Add power law exponent to legend
            plt.plot([], [], line_style, color=colors[i], label=f"Rep={rep}: y ∝ x^{b:.3f}")
    
    # Set log scales
    plt.xscale('log', base=2)
    if log_y:
        plt.yscale('log')
    
    # Add grid and labels
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("Offset", fontsize=12)
    plt.ylabel("Rouge-L Score", fontsize=12)
    
    # Add legend with improved styling
    # Move legend outside of the plot to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='white', edgecolor='black')
    plt.tight_layout()
    
    plt.show()


def compare_models(
    models: Dict[str, str],
    base_path: str,
    policy: str,
    offsets: List[int],
    prefixes: List[int],
    suffixes: List[int],
    metrics: Union[str, List[str]] = 'Rouge-L',
    offset: Optional[int] = None,
    prefix: Optional[int] = None,
    suffix: Optional[int] = None,
    repetition: int = 128,
    folder: str = 'sparse',
    show_std: bool = True
):
    """
    Compare metric statistics across different models with same policy.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping display names to model directory names
        e.g., {'1B': 'llama3-1b-15n-8192sl-60gbsz-standard', '3B': 'llama3-3b-...'}
    base_path : str
        Base path to results directory
    policy : str
        Decoding policy name (e.g., 'greedy', 'beam_nb5', 'nucleus')
    offsets, prefixes, suffixes : list
        Lists of parameter values used in the experiment
    metrics : str or list
        Single metric name or list of metrics to compare
    offset, prefix, suffix : int, optional
        Exactly two must be provided; the third varies
    repetition : int
        Which repetition to show (default: 128)
    folder : str
        Results folder name (default: 'sparse')
    show_std : bool
        Whether to show standard deviation (default: True)
    """
    if sum(x is not None for x in [offset, prefix, suffix]) != 2:
        raise ValueError("Exactly two parameters must be provided")
    
    # Import Results class if not already imported
    import sys
    sys.path.append('/iopsstor/scratch/cscs/xyixuan/PDM/src/verbatim_eval')
    from controlled_expr import Results
    
    # Construct file paths and load results
    results_dict = {}
    offsets_str = '_'.join(map(str, offsets))
    prefixes_str = '_'.join(map(str, prefixes))
    suffixes_str = '_'.join(map(str, suffixes))
    
    filename = f"offset_{offsets_str}_prefix_{prefixes_str}_suffix_{suffixes_str}_{policy}.pkl"
    
    for display_name, model_dir in models.items():
        path = f"{base_path}/{folder}/{model_dir}/{filename}"
        try:
            results_dict[display_name] = Results.load(path)
        except FileNotFoundError:
            print(f"Warning: Could not find file for model '{display_name}' at {path}")
            continue
    
    if not results_dict:
        print("No results files found!")
        return
    
    # Get first result to determine parameters
    first_result = next(iter(results_dict.values()))
    if offset is None:
        vary_by, iterations = 'offset', first_result.offsets
    elif prefix is None:
        vary_by, iterations = 'prefix', first_result.prefixes
    else:
        vary_by, iterations = 'suffix', first_result.suffixes
    
    fixed_vals = {
        'offset': offset,
        'prefix': prefix,
        'suffix': suffix
    }
    
    # Handle metrics parameter
    if isinstance(metrics, str):
        if metrics == 'all':
            metrics_list = first_result.metrics
        else:
            metrics_list = [metrics]
    else:
        metrics_list = metrics
    
    # Print comparison table
    print(f"\n=== Metric Comparison for Repetition {repetition} ===")
    print(f"Fixed: prefix={prefix}, suffix={suffix}" if offset is None else
          f"Fixed: offset={offset}, suffix={suffix}" if prefix is None else
          f"Fixed: offset={offset}, prefix={prefix}")
    print()
    
    # Define column widths based on show_std
    if show_std:
        value_width = 15  # "  0.965 ± 0.135"
        model_col_width = 15
    else:
        value_width = 9   # "  0.965"
        model_col_width = 9
    
    # Build header rows
    header1_parts = [f"{' ':>8} |"]
    header2_parts = [f"{vary_by.capitalize():>8} |"]
    
    for metric in metrics_list:
        # Calculate metric header span width
        # For each metric group: models are separated by " | "
        metric_group_width = len(models) * model_col_width + (len(models) - 1) * 3
        header1_parts.append(f" {metric:^{metric_group_width}} |")
        
        # Add model names
        for i, model_name in enumerate(models.keys()):
            if i == 0:
                header2_parts.append(f" {model_name:>{model_col_width}} ")
            else:
                header2_parts.append(f"| {model_name:>{model_col_width}} ")
        header2_parts.append("|")
    
    header1 = "".join(header1_parts)
    header2 = "".join(header2_parts)
    
    print(header1)
    print(header2)
    print("-" * len(header2))
    
    # Print data rows
    for val in iterations:
        current_offset = val if vary_by == 'offset' else fixed_vals['offset']
        current_prefix = val if vary_by == 'prefix' else fixed_vals['prefix']
        current_suffix = val if vary_by == 'suffix' else fixed_vals['suffix']
        
        row_parts = [f"{val:8} |"]
        
        for metric in metrics_list:
            for i, display_name in enumerate(models.keys()):
                if display_name in results_dict:
                    try:
                        stats = results_dict[display_name].get_stats(
                            results_dict[display_name].expr[0],
                            repetition,
                            current_offset,
                            current_prefix,
                            current_suffix,
                            metric
                        )
                        if show_std:
                            value = f"{stats.mean:6.3f} ± {stats.std:5.3f}"
                        else:
                            value = f"{stats.mean:6.3f}"
                        
                        if i == 0:
                            row_parts.append(f" {value:>{model_col_width}} ")
                        else:
                            row_parts.append(f"| {value:>{model_col_width}} ")
                    except:
                        if i == 0:
                            row_parts.append(f" {'N/A':>{model_col_width}} ")
                        else:
                            row_parts.append(f"| {'N/A':>{model_col_width}} ")
                else:
                    if i == 0:
                        row_parts.append(f" {'N/A':>{model_col_width}} ")
                    else:
                        row_parts.append(f"| {'N/A':>{model_col_width}} ")
            row_parts.append("|")
        
        print("".join(row_parts))


def plot_model_comparison(
    models: Dict[str, str],
    base_path: str,
    policy: str,
    offsets: List[int],
    prefixes: List[int],
    suffixes: List[int],
    metric: str = 'Rouge-L',
    offset: Optional[int] = None,
    prefix: Optional[int] = None,
    suffix: Optional[int] = None,
    repetition: int = 128,
    folder: str = 'sparse',
    log_scale: bool = True,
    show_std: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot metric comparison across different models with same policy.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping display names to model directory names
    base_path : str
        Base path to results directory
    policy : str
        Decoding policy name (e.g., 'greedy', 'beam_nb5', 'nucleus')
    offsets, prefixes, suffixes : list
        Lists of parameter values used in the experiment
    metric : str
        Metric to plot (default: 'Rouge-L')
    offset, prefix, suffix : int, optional
        Exactly two must be provided; the third varies
    repetition : int
        Which repetition to show (default: 128)
    folder : str
        Results folder name (default: 'sparse')
    log_scale : bool
        Whether to use log scale for x-axis (default: True)
    show_std : bool
        Whether to show error bars with standard deviation (default: True)
    figsize : tuple
        Figure size (default: (10, 6))
    """
    if sum(x is not None for x in [offset, prefix, suffix]) != 2:
        raise ValueError("Exactly two parameters must be provided")
    
    # Import Results class
    import sys
    sys.path.append('/iopsstor/scratch/cscs/xyixuan/PDM/src/verbatim_eval')
    from controlled_expr import Results
    
    # Load results
    results_dict = {}
    offsets_str = '_'.join(map(str, offsets))
    prefixes_str = '_'.join(map(str, prefixes))
    suffixes_str = '_'.join(map(str, suffixes))
    
    filename = f"offset_{offsets_str}_prefix_{prefixes_str}_suffix_{suffixes_str}_{policy}.pkl"
    
    for display_name, model_dir in models.items():
        path = f"{base_path}/{folder}/{model_dir}/{filename}"
        try:
            results_dict[display_name] = Results.load(path)
        except FileNotFoundError:
            print(f"Warning: Could not find file for model '{display_name}' at {path}")
            continue
    
    if not results_dict:
        print("No results files found!")
        return
    
    # Determine varying parameter
    first_result = next(iter(results_dict.values()))
    if offset is None:
        vary_by, iterations = 'offset', first_result.offsets
    elif prefix is None:
        vary_by, iterations = 'prefix', first_result.prefixes
    else:
        vary_by, iterations = 'suffix', first_result.suffixes
    
    fixed_vals = {
        'offset': offset,
        'prefix': prefix,
        'suffix': suffix
    }
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Plot each model
    for display_name, color in zip(models.keys(), colors):
        if display_name not in results_dict:
            continue
            
        x_values = []
        y_values = []
        y_errors = []
        
        for val in iterations:
            current_offset = val if vary_by == 'offset' else fixed_vals['offset']
            current_prefix = val if vary_by == 'prefix' else fixed_vals['prefix']
            current_suffix = val if vary_by == 'suffix' else fixed_vals['suffix']
            
            try:
                stats = results_dict[display_name].get_stats(
                    results_dict[display_name].expr[0],
                    repetition,
                    current_offset,
                    current_prefix,
                    current_suffix,
                    metric
                )
                x_values.append(val if val > 0 else 0.5)  # Avoid log(0)
                y_values.append(stats.mean)
                y_errors.append(stats.std)
            except:
                continue
        
        # Plot with or without error bars
        if show_std:
            plt.errorbar(x_values, y_values, yerr=y_errors, 
                        marker='o', markersize=8, linewidth=2,
                        label=display_name, color=color, capsize=5)
        else:
            plt.plot(x_values, y_values, 'o-', 
                    markersize=8, linewidth=2,
                    label=display_name, color=color)
    
    # Set scale
    if log_scale and vary_by == 'offset':
        plt.xscale('log', base=2)
    
    # Labels and title
    plt.xlabel(vary_by.capitalize(), fontsize=12)
    plt.ylabel(f'{metric} Score', fontsize=12)
    
    title = f"{metric} Comparison - {policy.upper()} (Rep={repetition})\n"
    title += f"Fixed: prefix={prefix}, suffix={suffix}" if offset is None else \
             f"Fixed: offset={offset}, suffix={suffix}" if prefix is None else \
             f"Fixed: offset={offset}, prefix={prefix}"
    plt.title(title, fontsize=14)
    
    # Grid and legend
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    
    # Set y-axis limits for Rouge-L
    if metric == 'Rouge-L':
        plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()

