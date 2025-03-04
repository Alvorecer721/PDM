import matplotlib.pyplot as plt
import numpy as np

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