from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .rouge_ttr import batch_rouge_ttr_calc
from .utils import load_inference_data


@dataclass
class EvalConfig:
    """Configuration for evaluation parameters"""
    base_path: str
    expr: str
    repetitions: np.ndarray
    policy: str
    vary_param: Literal['offset', 'prefix']  # What parameter to vary
    fixed_offset: int = 0  # Used when varying prefix length
    fixed_prefix_length: int = 500  # Used when varying offset
    len_suffix: int = 500
    offsets: Optional[List[int]] = field(default_factory=list)  # Optional when vary_param is 'prefix'
    prefix_lengths: Optional[List[int]] = field(default_factory=list)  # Optional when vary_param is 'prefix'

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.vary_param == 'offset' and not self.offsets:
            raise ValueError("offsets must be provided when vary_param is 'offset'")
        if self.vary_param == 'prefix' and not self.prefix_lengths:
            raise ValueError("prefix_lengths must be provided when vary_param is 'prefix'")


def eval_expr(config: EvalConfig) -> Dict:
    """
    Evaluate expressions by either offset or prefix length
    """
    results = {}
    results[config.expr] = {}

    pbar1 = tqdm(config.repetitions, desc="Processing repetitions")
    
    for r in pbar1:
        pbar1.set_description(f"Processing repetition {r}")
        results[config.expr][r] = {}

        # Determine what to iterate over based on vary_param
        if config.vary_param == 'offset':
            iter_values = config.offsets
            fixed_prefix = config.fixed_prefix_length
            desc = "Processing offsets"
        else:  # prefix
            iter_values = config.prefix_lengths
            fixed_offset = config.fixed_offset
            desc = "Processing prefix lengths"

        pbar2 = tqdm(iter_values, desc=desc, leave=False)
        for val in pbar2:
            if config.vary_param == 'offset':
                offset = val
                prefix_length = fixed_prefix
                pbar2.set_description(f"Processing offset {val}")
            else:
                offset = fixed_offset
                prefix_length = val
                pbar2.set_description(f"Processing prefix length {val}")
            
            try:
                data_path = f"{config.base_path}/{config.expr}/inference"
                data = load_inference_data(
                    data_path, 
                    rep=r,
                    policy=config.policy,
                    offset=offset,
                    len_prefix=prefix_length,
                    len_suffix=config.len_suffix
                )

                # Calculate metrics
                eval_results = data.map(
                    batch_rouge_ttr_calc,
                    batched=True, 
                    batch_size=5,
                    num_proc=100,
                    desc=f"Calculating metrics for rep={r}, {config.vary_param}={val}",
                    remove_columns=data.column_names,
                    fn_kwargs={
                        "true_key": "true_suffix",
                        "gen_key": "generated_suffix",
                        "len_suffix": config.len_suffix,
                    }
                )

                # Store results using the varying parameter as key
                key = val
                results[config.expr][r][key] = {}

                # NLL
                results[config.expr][r][key]['NLL'] = {
                    'scores': data['nll_mean'],
                    'mean': np.mean(data['nll_mean']),
                    'std': np.std(data['nll_mean'])
                }

                # Store other metrics
                for metric in eval_results.column_names:
                    scores = np.array(eval_results[metric])
                    results[config.expr][r][key][metric] = {
                        'scores': scores,
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    }
            except Exception as e:
                print(f"\nError processing rep={r}, {config.vary_param}={val}: {str(e)}")
                continue

        pbar2.close()
    
    pbar1.close()
    return results


def format_metric_df(result, metric, vary_param='offset', stat='mean'):
    """
    Convert experiment results into a pandas DataFrame.
    Args:
        result: Dictionary containing a single experiment's results
        metric: String specifying which metric to show
        vary_param: String indicating what parameter was varied ('offset' or 'prefix')
        stat: Which statistics to include ('all', 'mean', or 'std')
    Returns:
        DataFrame with varied parameter as rows and repetition as columns,
        maintaining proper naming for all formats
    """
    # Get the only experiment's data
    rep_dict = next(iter(result.values()))
    
    # Get all unique values and repetitions 
    values = sorted(set(val for rep_d in rep_dict.values() for val in rep_d.keys()))
    repetitions = sorted(rep_dict.keys())
    
    if stat == 'all':
        # Create MultiIndex for columns with both mean and std
        columns = pd.MultiIndex.from_product([
            repetitions,
            ['mean', 'std']
        ], names=['repetition', 'stat'])
        
        # Initialize DataFrame with NaN values
        df = pd.DataFrame(index=values, columns=columns)
        
        # Fill the DataFrame with both mean and std
        for rep, val_dict in rep_dict.items():
            for val, metrics_dict in val_dict.items():
                if metric in metrics_dict:
                    df.loc[val, (rep, 'mean')] = metrics_dict[metric]['mean']
                    df.loc[val, (rep, 'std')] = metrics_dict[metric]['std']
    
    else:
        # Create named columns for single stat
        columns = pd.Index(repetitions, name='repetition')
        df = pd.DataFrame(index=values, columns=columns)
        
        # Fill the DataFrame with only the requested stat
        for rep, val_dict in rep_dict.items():
            for val, metrics_dict in val_dict.items():
                if metric in metrics_dict:
                    df.loc[val, rep] = metrics_dict[metric][stat]
    
    # Set index name based on vary_param
    df.index = pd.Index(values, name='prefix_length' if vary_param == 'prefix' else 'offset')
    return df


def print_metrics(results: Dict, metric=None, value=None, vary_param='offset'):
    """
    Log metrics summary for experiment with repetitions and varied parameter.
    Args:
        results: Dictionary output from eval_expr
        metric: Optional string to filter for a specific metric
        value: Optional int to filter for a specific offset/prefix length
        vary_param: String indicating what parameter was varied ('offset' or 'prefix')
    """
    value_name = 'Prefix Length' if vary_param == 'prefix' else 'Offset'
    
    for expr, rep_dict in results.items():
        print(f"\n=== Summary for Experiment: {expr} ===")
        for rep, val_dict in rep_dict.items():
            print(f"\n === Summary for Repetition {rep:3d} ===")
            for curr_val, metrics_dict in val_dict.items():
                # Skip if value is specified and doesn't match
                if value is not None and curr_val != value:
                    continue
                    
                print(f" {value_name} {curr_val:3d}")
                if metric:
                    if metric in metrics_dict:
                        stats = metrics_dict[metric]
                        print(
                            f" {metric:<10} | "
                            f"Mean = {stats['mean']:.3f}, "
                            f"Std = {stats['std']:.3f}"
                        )
                else:
                    for m, stats in metrics_dict.items():
                        # Skip printing the raw scores
                        if m != 'scores':
                            print(
                                f" {m:<10} | "
                                f"Mean = {stats['mean']:.3f}, "
                                f"Std = {stats['std']:.3f}"
                            )