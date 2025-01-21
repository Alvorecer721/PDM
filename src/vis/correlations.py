import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlations(scores1, scores2):
    """Calculate Pearson and Spearman correlations between two sets of scores."""
    pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
    spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
    return {
        'pearson': {'r': pearson_r, 'p': pearson_p},
        'spearman': {'rho': spearman_r, 'p': spearman_p}
    }

def plot_pearson_correlation(result, expr, metric1, metric2):
    """Plot Pearson correlation between two metrics for all repetitions.
    
    Args:
        result: Dictionary containing the results
        expr: Name of the experiment
        metric1: Name of first metric (x-axis)
        metric2: Name of second metric (y-axis)

    Example Usage:
    >>> plot_pearson_correlation(
    >>>     goldfish_res_greedy, 
    >>>     expr='5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_1984000',
    >>>     metric1='TTR_gen', metric2='Rouge-L'
    >>> )
    """
    reps = sorted(result[expr].keys())
    n_reps = len(reps)
    n_cols = 3
    n_rows = (n_reps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_reps > 1 else [axes]
    
    print(f"Pearson Correlation Tests between {metric1} and {metric2}:")
    
    for idx, rep in enumerate(reps):
        scores1 = result[expr][rep][metric1]['scores']
        scores2 = result[expr][rep][metric2]['scores']
        
        pearson_r, pearson_p = stats.pearsonr(scores1, scores2)
        
        ax = axes[idx]
        sns.scatterplot(x=scores1, y=scores2, ax=ax)
        sns.regplot(x=scores1, y=scores2, scatter=False, color='red', ax=ax)
        
        ax.set_xlabel(f'{metric1} Scores')
        ax.set_ylabel(f'{metric2} Scores')
        ax.set_title(f'Repetition {rep}')
        
        stats_text = f'r = {pearson_r:.4f}\np = {pearson_p:.4f}'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        print(f"\nRepetition {rep}:")
        print(f"r = {pearson_r:.4f}")
        print(f"p-value = {pearson_p:.4f}")
    
    for idx in range(len(reps), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f'Pearson Correlation between {metric1} and {metric2}\n{expr}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_spearman_correlation(result, expr, metric1, metric2):
    """Plot Spearman correlation between two metrics for all repetitions.
    
    Args:
        result: Dictionary containing the results
        expr: Name of the experiment
        metric1: Name of first metric (x-axis)
        metric2: Name of second metric (y-axis)

    Example Usage:
    >>> plot_spearman_correlation(
    >>>     goldfish_res_greedy, 
    >>>     expr='5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_1984000',
    >>>     metric1='TTR_gen', metric2='Rouge-L'
    >>> )
    """
    reps = sorted(result[expr].keys())
    n_reps = len(reps)
    n_cols = 3
    n_rows = (n_reps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_reps > 1 else [axes]
    
    print(f"Spearman Correlation Tests between {metric1} and {metric2}:")
    
    for idx, rep in enumerate(reps):
        scores1 = np.array(result[expr][rep][metric1]['scores'])
        scores2 = np.array(result[expr][rep][metric2]['scores'])
        
        # Convert to ranks for Spearman visualization
        ranks1 = stats.rankdata(scores1)
        ranks2 = stats.rankdata(scores2)
        
        spearman_r, spearman_p = stats.spearmanr(scores1, scores2)
        
        ax = axes[idx]
        sns.scatterplot(x=ranks1, y=ranks2, ax=ax)
        sns.regplot(x=ranks1, y=ranks2, scatter=False, color='red', ax=ax)
        
        ax.set_xlabel(f'{metric1} Ranks')
        ax.set_ylabel(f'{metric2} Ranks')
        ax.set_title(f'Repetition {rep}')
        
        stats_text = f'rho = {spearman_r:.4f}\np = {spearman_p:.4f}'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        print(f"\nRepetition {rep}:")
        print(f"rho = {spearman_r:.4f}")
        print(f"p-value = {spearman_p:.4f}")
    
    for idx in range(len(reps), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f'Spearman Correlation between {metric1} and {metric2}\n{expr}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()