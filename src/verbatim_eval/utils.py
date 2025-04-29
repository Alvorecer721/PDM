from pathlib import Path
from datasets import load_dataset
import numpy as np
import pandas as pd
import os
import json
from typing import Union, List, Dict, Optional

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
    try:
        dataset = load_dataset(
            'json', 
            data_files=[str(f) for f in rank_files], 
            split='train',
            cache_dir="/iopsstor/scratch/cscs/xyixuan/gutenberg"
        )
    except ValueError as e:
        print(f"Failed to load dataset from path: {file_path}")
        raise e
    
    total_size = len(dataset)
    items_per_rank = total_size // world_size

    # Reorder to original sequential order
    orig_indices = np.arange(total_size).reshape(world_size, items_per_rank).T.flatten()
    
    return dataset.select(orig_indices)

def load_mauve_scores(parent_dir: str, 
                      reps: Optional[List[int]] = None, 
                      prefixes: Optional[List[int]] = None, 
                      suffixes: Optional[List[int]] = None, 
                      offsets: Optional[List[int]] = None,
                      as_series: bool = True) -> Union[Dict, pd.DataFrame, pd.Series]:
    """
    Load MAUVE scores and return them in the specified format.
    
    Parameters:
        parent_dir (str): Path to the parent directory containing subdirectories
        reps (list): List of repetition numbers to retrieve (e.g., [32, 64, 128])
        prefixes (list): List of prefix values to match (e.g., [50, 100, 250])
        suffixes (list): List of suffix values to match (e.g., [500])
        offsets (list): List of offset values to match (e.g., [0, 1, 2])
        as_series (bool): If True, returns a pandas Series when there's a single offset/prefix/suffix
                         If False, always returns the nested dictionary
    
    Returns:
        Union[dict, pd.DataFrame, pd.Series]: 
            - pd.Series with rep as index when as_series=True and single offset/prefix/suffix
            - pd.DataFrame when as_series=True but multiple offset/prefix/suffix combinations
            - Nested dictionary with structure {offset: {prefix: {suffix: {rep: mauve_score}}}}
              when as_series=False
    """
    # Default to all if not specified
    reps = reps or []
    prefixes = prefixes or []
    suffixes = suffixes or []
    offsets = offsets or []
    
    results = {}
    
    # Iterate over all subdirectories in the parent directory
    for offset_dir in os.listdir(parent_dir):
        offset_dir_path = os.path.join(parent_dir, offset_dir)
        
        # Check if it's a directory and matches our pattern for offsets
        if os.path.isdir(offset_dir_path) and offset_dir.startswith("offset_"):
            try:
                # Extract offset value
                offset_val = int(offset_dir.replace('offset_', ''))
                
                # Skip if not in our target offsets (when specified)
                if offsets and offset_val not in offsets:
                    continue
                
                # Initialize offset in results
                if offset_val not in results:
                    results[offset_val] = {}
                
                # Process prefix_suffix directories
                for prefix_suffix_dir in os.listdir(offset_dir_path):
                    prefix_suffix_path = os.path.join(offset_dir_path, prefix_suffix_dir)
                    
                    if os.path.isdir(prefix_suffix_path) and prefix_suffix_dir.startswith("prefix_"):
                        # Extract prefix and suffix from directory name
                        parts = prefix_suffix_dir.split('_')
                        if len(parts) >= 4:  # Ensure we have enough parts
                            try:
                                prefix_val = int(parts[1])
                                suffix_val = int(parts[3])
                                
                                # Skip if not in our target lists (when specified)
                                if prefixes and prefix_val not in prefixes:
                                    continue
                                if suffixes and suffix_val not in suffixes:
                                    continue
                                
                                # Initialize nested dictionaries
                                if prefix_val not in results[offset_val]:
                                    results[offset_val][prefix_val] = {}
                                if suffix_val not in results[offset_val][prefix_val]:
                                    results[offset_val][prefix_val][suffix_val] = {}
                                
                                # Process rep files
                                for filename in os.listdir(prefix_suffix_path):
                                    if filename.startswith("rep_") and filename.endswith(".json"):
                                        try:
                                            rep_num = int(filename.replace('rep_', '').replace('.json', ''))
                                            
                                            # Skip if not in our target reps (when specified)
                                            if reps and rep_num not in reps:
                                                continue
                                            
                                            # Load the json file
                                            file_path = os.path.join(prefix_suffix_path, filename)
                                            with open(file_path, 'r') as f:
                                                data = json.load(f)
                                            
                                            if 'mauve_score' in data:
                                                results[offset_val][prefix_val][suffix_val][rep_num] = data['mauve_score']
                                        except (ValueError, json.JSONDecodeError):
                                            # Skip files with invalid rep format or JSON issues
                                            continue
                            
                            except (ValueError, IndexError):
                                # Skip directories with invalid format
                                continue
            
            except ValueError:
                # Skip directories with invalid offset format
                continue
    
    # Sort the dictionaries by keys for consistent output
    sorted_results = {}
    for offset in sorted(results.keys()):
        sorted_results[offset] = {}
        for prefix in sorted(results[offset].keys()):
            sorted_results[offset][prefix] = {}
            for suffix in sorted(results[offset][prefix].keys()):
                sorted_results[offset][prefix][suffix] = dict(sorted(results[offset][prefix][suffix].items()))
    
    # If not requesting pandas format, return the nested dictionary
    if not as_series:
        return sorted_results
    
    # For pandas output, first check if we have data
    if not sorted_results:
        return pd.DataFrame() if len(results) != 1 else pd.Series(name="mauve")
    
    # Handle the common case: single offset, prefix, suffix combination
    if len(sorted_results) == 1:
        offset = list(sorted_results.keys())[0]
        if len(sorted_results[offset]) == 1:
            prefix = list(sorted_results[offset].keys())[0]
            if len(sorted_results[offset][prefix]) == 1:
                suffix = list(sorted_results[offset][prefix].keys())[0]
                # Convert to pandas Series with proper index
                scores_dict = sorted_results[offset][prefix][suffix]
                return pd.Series(scores_dict, name="mauve")
    
    # For multiple combinations, return a DataFrame
    records = []
    for offset, prefix_dict in sorted_results.items():
        for prefix, suffix_dict in prefix_dict.items():
            for suffix, rep_dict in suffix_dict.items():
                for rep, score in rep_dict.items():
                    records.append({
                        'offset': offset,
                        'prefix': prefix,
                        'suffix': suffix,
                        'rep': rep,
                        'mauve': score
                    })
    
    return pd.DataFrame(records) 


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