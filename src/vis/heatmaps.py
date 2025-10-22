import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_heatmaps_subplots(data_dict, output_file, figsize=(20, 6), dpi=300):
    """
    Create heatmaps from a dictionary of pandas dataframes with fixed colorbar scale 0-1
    LaTeX-ready formatting with professional fonts and spacing

    Args:
        data_dict: Dictionary with keys as titles and values as pandas dataframes
        output_file: Output filename for the plot
        figsize: Tuple for figure size (width, height) - optimized for LaTeX text width
        dpi: DPI for saving the figure (higher = better quality when scaled)
    """
    # Calculate number of rows needed
    n_plots = len(data_dict)
    n_cols = min(3, n_plots)  # At most 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots grid with more spacing to prevent overlapping
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows),
                           gridspec_kw={'wspace': 0.15, 'hspace': 0.25})

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

        # Create custom annotation format without leading zero
        annot_data = []
        for row in data:
            formatted_row = []
            for val in row:
                formatted = f'{val:.3f}'
                # Remove leading zero for values between 0 and 1
                if 0 < val < 1:
                    formatted = formatted.replace('0.', '.')
                formatted_row.append(formatted)
            annot_data.append(formatted_row)

        # Plot heatmap with fixed scale
        sns.heatmap(data, annot=annot_data, fmt='', cmap='YlOrRd', ax=ax,
                    xticklabels=df.columns, yticklabels=df.index,
                    annot_kws={'size': 10},  # Cell values
                    vmin=vmin, vmax=vmax)

        ax.set_title(title, fontsize=14, pad=10, weight='bold')  # Title
        ax.set_xlabel(xlabel, fontsize=12, weight='bold')  # X-axis label
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')  # Y-axis label

        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=10, width=1.5)
        ax.tick_params(axis='y', labelsize=10, width=1.5)

        # Make colorbar tick labels larger
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=10)

    # Hide empty subplots
    for idx in range(len(items), n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    plt.savefig(f"/iopsstor/scratch/cscs/xyixuan/PDM/results/plots/{output_file}.pdf",
                bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    # plt.tight_layout()

def create_heatmaps_difference_subplots(data_dict1, data_dict2, output_file, figsize=(20, 6), vmin=None, vmax=None, dpi=300):
    """
    Create heatmaps showing the difference between two dictionaries of pandas dataframes
    data_dict1 - data_dict2

    Args:
        data_dict1: First dictionary with keys as titles and values as pandas dataframes
        data_dict2: Second dictionary with keys as titles and values as pandas dataframes
        output_file: Output filename for the plot
        figsize: Tuple for figure size (width, height) - optimized for LaTeX text width
        vmin: Minimum value for the colorbar (optional)
        vmax: Maximum value for the colorbar (optional)
        dpi: DPI for saving the figure (higher = better quality when scaled)
    """
    # Verify matching keys
    if set(data_dict1.keys()) != set(data_dict2.keys()):
        raise ValueError("The two dictionaries must have the same keys")

    # Create difference dictionary and filter out unwanted repetitions and prefix lengths
    diff_dict = {}
    reps_to_exclude = [24, 48, 96]
    prefix_lengths_to_exclude = [1500]  # Exclude prefix length 1500

    for key in data_dict1.keys():
        if not data_dict1[key].equals(data_dict2[key].reindex_like(data_dict1[key])):
            diff_df = data_dict1[key] - data_dict2[key].reindex_like(data_dict1[key])

            # Filter out unwanted repetitions from index if they exist
            existing_reps_to_exclude = [rep for rep in reps_to_exclude if rep in diff_df.index]
            if existing_reps_to_exclude:
                diff_df = diff_df.drop(existing_reps_to_exclude)

            # Filter out unwanted prefix lengths from columns if they exist
            existing_prefix_to_exclude = [pl for pl in prefix_lengths_to_exclude if pl in diff_df.columns]
            if existing_prefix_to_exclude:
                diff_df = diff_df.drop(columns=existing_prefix_to_exclude)

            diff_dict[f'Difference ({key})'] = diff_df

    # Calculate number of rows needed
    n_plots = len(diff_dict)
    if n_plots == 0:
        print("No differences found between the dictionaries")
        return

    n_cols = min(3, n_plots)  # At most 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots grid with more spacing to prevent overlapping
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows),
                           gridspec_kw={'wspace': 0.15, 'hspace': 0.25})
    
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
        
        # Create custom annotation format without leading zero
        annot_data = []
        for row in data:
            formatted_row = []
            for val in row:
                formatted = f'{val:.3f}'
                # Remove leading zero for values between -1 and 1
                if -1 < val < 0:
                    formatted = formatted.replace('-0.', '-.')
                elif 0 < val < 1:
                    formatted = formatted.replace('0.', '.')
                formatted_row.append(formatted)
            annot_data.append(formatted_row)

        # Plot heatmap with specified scale and diverging colormap
        sns.heatmap(data, annot=annot_data, fmt='', cmap='RdBu_r', ax=ax,
                    xticklabels=df.columns, yticklabels=df.index,
                    annot_kws={'size': 10},  # Cell values
                    vmin=vmin, vmax=vmax,
                    center=0)

        ax.set_title(title, fontsize=14, pad=10, weight='bold')  # Title
        ax.set_xlabel(xlabel, fontsize=12, weight='bold')  # X-axis label
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')  # Y-axis label

        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=10, width=1.5)
        ax.tick_params(axis='y', labelsize=10, width=1.5)

        # Make colorbar tick labels larger
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=10)
    
    # Hide empty subplots
    for idx in range(len(items), n_rows * n_cols):
        axes.flat[idx].set_visible(False)
    
    plt.savefig(f"/iopsstor/scratch/cscs/xyixuan/PDM/results/plots/{output_file}.pdf",
                bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    # plt.tight_layout()