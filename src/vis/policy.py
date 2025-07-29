import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional


def plot_policy_comparison(
    policies: List[str],
    base_path: str,
    expr: str,
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
    Plot metric comparison across different policies for the same model.
    
    Parameters
    ----------
    policies : list
        List of policy names to compare
    base_path : str
        Base path to results directory
    expr : str
        Experiment/model name
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
    
    for policy in policies:
        filename = f"offset_{offsets_str}_prefix_{prefixes_str}_suffix_{suffixes_str}_{policy}.pkl"
        path = f"{base_path}/{folder}/{expr}/{filename}"
        try:
            results_dict[policy] = Results.load(path)
        except FileNotFoundError:
            print(f"Warning: Could not find file for policy '{policy}' at {path}")
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
    
    # Colors for different policies
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    
    # Plot each policy
    for policy, color in zip(policies, colors):
        if policy not in results_dict:
            continue
            
        x_values = []
        y_values = []
        y_errors = []
        
        for val in iterations:
            current_offset = val if vary_by == 'offset' else fixed_vals['offset']
            current_prefix = val if vary_by == 'prefix' else fixed_vals['prefix']
            current_suffix = val if vary_by == 'suffix' else fixed_vals['suffix']
            
            try:
                stats = results_dict[policy].get_stats(
                    results_dict[policy].expr[0],
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
                        label=policy, color=color, capsize=5)
        else:
            plt.plot(x_values, y_values, 'o-', 
                    markersize=8, linewidth=2,
                    label=policy, color=color)
    
    # Set scale
    if log_scale and vary_by == 'offset':
        plt.xscale('log', base=2)
    
    # Labels and title
    plt.xlabel(vary_by.capitalize(), fontsize=12)
    plt.ylabel(f'{metric} Score', fontsize=12)
    
    title = f"{metric} Comparison - {expr} (Rep={repetition})\n"
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


def compare_policies(
    policies: List[str],
    base_path: str,
    expr: str,
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
    Compare metric statistics across different decoding policies.
    
    Parameters
    ----------
    policies : list
        List of policy names to compare
        e.g., ['greedy', 'beam_nb5', 'nucleus']
    base_path : str
        Base path to results directory
    expr : str
        Experiment name
    offsets, prefixes, suffixes : list
        Lists of parameter values used in the experiment
    metrics : str or list
        Single metric name or list of metrics to compare
        If 'all', shows all available metrics
    offset, prefix, suffix : int, optional
        Exactly two must be provided; the third varies
    repetition : int
        Which repetition to show (default: 128)
    folder : str
        Results folder name (default: 'sparse')
    show_std : bool
        Whether to show standard deviation (default: True)
    """
    # Import the function from controlled_expr
    import sys
    sys.path.append('/iopsstor/scratch/cscs/xyixuan/PDM/src/verbatim_eval')
    from controlled_expr import compare_policies as compare_policies_impl
    
    # Call the implementation
    compare_policies_impl(
        policies=policies,
        base_path=base_path,
        expr=expr,
        offsets=offsets,
        prefixes=prefixes,
        suffixes=suffixes,
        metrics=metrics,
        offset=offset,
        prefix=prefix,
        suffix=suffix,
        repetition=repetition,
        folder=folder,
        show_std=show_std
    )