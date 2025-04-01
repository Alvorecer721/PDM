import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_heatmaps_subplots(data_dict, output_file, figsize=(40, 10)):
    """
    Create heatmaps from a dictionary of pandas dataframes with fixed colorbar scale 0-1
    
    Args:
        data_dict: Dictionary with keys as titles and values as pandas dataframes
        figsize: Tuple for figure size (width, height)
    """
    # Calculate number of rows needed
    n_plots = len(data_dict)
    n_cols = min(3, n_plots)  # At most 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    
    # Handle single plot case
    if n_plots == 1:
        axes = np.array([[axes]])
    # Convert axes to 2D array if it's 1D
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Get the first dataframe to extract column and index names
    first_df = next(iter(data_dict.values()))
    xlabel = first_df.columns.name
    ylabel = first_df.index.name
    
    # Set fixed color scale from 0 to 1
    vmin, vmax = 0, 1
    
    # Create heatmap for each dataframe
    items = list(data_dict.items())
    for idx, ((title, df), ax) in enumerate(zip(items, axes.flat)):
        # Convert to numpy array and ensure float type
        data = df.astype(float).to_numpy()
        
        # Plot heatmap with fixed scale
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                    xticklabels=df.columns, yticklabels=df.index,
                    annot_kws={'size': 8},
                    vmin=vmin, vmax=vmax)
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Hide empty subplots
    for idx in range(len(items), n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    plt.savefig(f"/capstor/users/cscs/xyixuan/PDM/results/plots/{output_file}.pdf", bbox_inches='tight', pad_inches=0.1)
    
    # plt.tight_layout()

def create_heatmaps_difference_subplots(data_dict1, data_dict2, output_file, figsize=(30, 12), vmin=None, vmax=None):
    """
    Create heatmaps showing the difference between two dictionaries of pandas dataframes
    
    Args:
        data_dict1: First dictionary with keys as titles and values as pandas dataframes
        data_dict2: Second dictionary with keys as titles and values as pandas dataframes
        figsize: Tuple for figure size (width, height)
        vmin: Minimum value for the colorbar (optional)
        vmax: Maximum value for the colorbar (optional)
    """
    # Verify matching keys
    if set(data_dict1.keys()) != set(data_dict2.keys()):
        raise ValueError("The two dictionaries must have the same keys")
    
    # Create difference dictionary
    diff_dict = {}
    for key in data_dict1.keys():
        if not data_dict1[key].equals(data_dict2[key].reindex_like(data_dict1[key])):
            diff_dict[f'Difference ({key})'] = data_dict1[key] - data_dict2[key].reindex_like(data_dict1[key])
    
    # Calculate number of rows needed
    n_plots = len(diff_dict)
    if n_plots == 0:
        print("No differences found between the dictionaries")
        return
        
    n_cols = min(3, n_plots)  # At most 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    
    # Handle single plot case
    if n_plots == 1:
        axes = np.array([[axes]])
    # Convert axes to 2D array if it's 1D
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Get the first dataframe to extract column and index names
    first_df = next(iter(diff_dict.values()))
    xlabel = first_df.columns.name
    ylabel = first_df.index.name
    
    # If vmin and vmax are not provided, calculate them from the data
    if vmin is None or vmax is None:
        all_values = np.concatenate([df.values.flatten() for df in diff_dict.values()])
        abs_max = max(abs(np.min(all_values)), abs(np.max(all_values)))
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    
    # Create heatmap for each difference dataframe
    items = list(diff_dict.items())
    for idx, ((title, df), ax) in enumerate(zip(items, axes.flat)):
        # Convert to numpy array and ensure float type
        data = df.astype(float).to_numpy()
        
        # Plot heatmap with specified scale and diverging colormap
        sns.heatmap(data, annot=True, fmt='.4f', cmap='RdBu_r', ax=ax,
                    xticklabels=df.columns, yticklabels=df.index,
                    annot_kws={'size': 8},
                    vmin=vmin, vmax=vmax,
                    center=0)
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Hide empty subplots
    for idx in range(len(items), n_rows * n_cols):
        axes.flat[idx].set_visible(False)
    
    plt.savefig(f"/capstor/users/cscs/xyixuan/PDM/results/plots/{output_file}.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()