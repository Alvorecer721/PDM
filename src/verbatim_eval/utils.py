from pathlib import Path
from datasets import load_dataset
import numpy as np
import pandas as pd

def load_inference_data(base_dir, step=None, consumed=None, offset=None, len_prefix=None, len_suffix=None, rep=None, policy=None):
    """
    Load inference data from a given step and restore original data order.
    DistributedSampler distributes data in round-robin fashion (e.g., GPU0: [0,8,16,...], GPU1: [1,9,17,...]).
    This function loads data from all rank files and reorders it back to sequential order [0,1,2,...].
    
    Parameters
    ----------
    base_dir : str
        Base directory path 

    For dense experiments:

    step : int
        Training step
    consumed : int
        Number of consumed tokens
    
    For sparse experiments:
    rep : int
        Repetition number
        
    Returns
    -------
    Dataset
        HuggingFace dataset with data restored to original sequential order
    """

    assert (step is not None and consumed is not None) or rep is not None, "Either step and consumed or rep must be provided"
    assert (step is None and consumed is None) or rep is None, "Either step and consumed or rep must be provided, not both"

    if (rep is not None) and (policy is not None):
        file_path = Path(base_dir) / f"offset_{offset}_prefix_{len_prefix}_suffix_{len_suffix}" / f"rep_{rep}_{policy}"
    else:
        file_path = Path(base_dir) / f"step={step}-consumed={consumed}"
    
    rank_files = sorted(file_path.glob("rank*.jsonl"))
    world_size = len(rank_files)

    # Load data from all ranks
    dataset = load_dataset(
        'json', 
        data_files=[str(f) for f in rank_files], 
        split='train'
    )
    
    total_size = len(dataset)
    items_per_rank = total_size // world_size

    # Reorder to original sequential order
    orig_indices = np.arange(total_size).reshape(world_size, items_per_rank).T.flatten()
    
    return dataset.select(orig_indices)


def find_top_quantile_indices(metric_1_scores, metric_2_scores, q=0.1):
    """
    Find indices that are in the top q percentile for both metrics.
    
    Parameters
    ----------
    metric_1_scores : np.ndarray, scores for metric 1
    metric_2_scores : np.ndarray, scores for metric 2
    q : float, top q percentile
    """

    assert len(metric_1_scores) == len(metric_2_scores), "Metric scores must be consistent"

    # Calculate the thresholds for top q percentile
    metric_1_threshold = np.quantile(metric_1_scores, 1 - q)
    metric_2_threshold = np.quantile(metric_2_scores, 1 - q)
    
    # Find indices that meet both criteria
    indices = [i for i in range(len(metric_1_scores)) 
              if metric_1_scores[i] >= metric_1_threshold and metric_2_scores[i] >= metric_2_threshold]
    
    return indices


def collect_experiment_data(results, required_metric=None):
    """
    Collect and combine experimental results into a single DataFrame.
    
    Parameters:
    -----------
    results : dict
        Nested dictionary containing experimental results
    required_metric : str, optional
        If specified, only collect data from experiments containing this metric
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame containing all experimental data
    
    Raises:
    -------
    ValueError
        If required_metric is specified but no data contains that metric
    """
    all_data = []
    
    for exp in results:
        for rep in results[exp]:
            # Check if required metric exists when specified
            if required_metric is not None and required_metric not in results[exp][rep]:
                continue
                
            data = {metric: results[exp][rep][metric]['scores'] 
                   for metric in results[exp][rep]}
            df = pd.DataFrame(data)
            df['exp'] = exp
            df['rep'] = rep
            all_data.append(df)
    
    if not all_data:
        if required_metric:
            raise ValueError(f"No data found containing metric: {required_metric}")
        else:
            raise ValueError("No data found in results")
            
    return pd.concat(all_data)


def threshold_metrics(results, threshold_metric, threshold_value, threshold_operator='>='):
    """
    Filter and aggregate metrics based on a threshold condition for any specified metric.
    
    Parameters:
    -----------
    results : dict
        Nested dictionary containing experimental results
    threshold_metric : str
        Name of the metric to use for thresholding
    threshold_value : float
        Value to threshold against
    threshold_operator : str, optional
        Comparison operator to use ('>=', '>', '<=', '<', '==')
        Default is '>='
        
    Returns:
    --------
    tuple
        (thresholded_aggregated_data, full_dataframe)
    """
    operators = {
        '>=': lambda x, y: x >= y,
        '>': lambda x, y: x > y,
        '<=': lambda x, y: x <= y,
        '<': lambda x, y: x < y,
        '==': lambda x, y: x == y
    }
    
    if threshold_operator not in operators:
        raise ValueError(f"Invalid operator. Must be one of: {', '.join(operators.keys())}")
    
    # Collect and combine all data
    df = collect_experiment_data(results, required_metric=threshold_metric)
    
    # Apply threshold
    mask = operators[threshold_operator](df[threshold_metric], threshold_value)
    thresholded = df[mask].groupby(['exp', 'rep']).agg(['mean', 'std', 'count'])
    
    return thresholded, df