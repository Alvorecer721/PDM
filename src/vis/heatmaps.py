import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_heatmaps_subplots(data_dict, figsize=(30, 12)):
    """
    Create heatmaps from a dictionary of pandas dataframes
    
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
    
    fig.suptitle('Rouge-L Scores by Offset, Prefix Length, and Repetition', fontsize=16, y=1.02)
    
    # Get the first dataframe to extract column and index names
    first_df = next(iter(data_dict.values()))
    xlabel = first_df.columns.name
    ylabel = first_df.index.name
    
    # Create heatmap for each dataframe
    items = list(data_dict.items())
    for idx, ((title, df), ax) in enumerate(zip(items, axes.flat)):
        # Convert to numpy array and ensure float type
        data = df.astype(float).to_numpy()
        
        # Plot heatmap
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                    xticklabels=df.columns, yticklabels=df.index,
                    annot_kws={'size': 8})
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Hide empty subplots
    for idx in range(len(items), n_rows * n_cols):
        axes.flat[idx].set_visible(False)
    
    plt.tight_layout()