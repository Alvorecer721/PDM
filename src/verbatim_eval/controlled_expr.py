from typing import List, Dict, Any, Literal, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
import os

try:
    from .rouge_ttr import batch_rouge_ttr_calc
    from .utils import load_inference_data
except ImportError:
    from rouge_ttr import batch_rouge_ttr_calc
    from utils import load_inference_data

@dataclass
class MetricData:
    scores: np.ndarray
    mean: float
    std: float


@dataclass 
class MetricData:
   scores: np.ndarray
   mean: float 
   std: float

@dataclass
class Results:
    data: Dict
    policy: Optional[str] = None

    def _get_nested_keys(self, level: int) -> List:
        d = self.data
        for _ in range(level):
            d = d[next(iter(d))]
        return sorted(d.keys())

    @property
    def expr(self) -> List[str]: 
        return self._get_nested_keys(0)

    @property
    def repetitions(self) -> List[int]:
        return [int(x) for x in self._get_nested_keys(1)]
        
    @property
    def offsets(self) -> List[int]:
        return [int(x) for x in self._get_nested_keys(2)]
        
    @property
    def prefixes(self) -> List[int]:
        return [int(x) for x in self._get_nested_keys(3)]
        
    @property
    def suffixes(self) -> List[int]:
        return [int(x) for x in self._get_nested_keys(4)]
        
    @property
    def metrics(self) -> List[str]:
        return self._get_nested_keys(5)

    @staticmethod
    def _convert_to_metric_data(d: Dict) -> Union[Dict, MetricData]:
        # If it's already a MetricData object, return it as-is
        if isinstance(d, MetricData):
            return d
        
        # If it's a dictionary with all required keys, convert to MetricData
        if isinstance(d, dict) and all(k in d for k in ['scores', 'mean', 'std']):
            return MetricData(d['scores'], d['mean'], d['std'])
        
        # If it's a dictionary, recursively convert its values
        if isinstance(d, dict):
            return {k: Results._convert_to_metric_data(v) for k, v in d.items()}
        
        # For any other type, return it as-is
        return d

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Extract policy from filename if available
            filename = os.path.basename(path)
            parts = filename.replace('.pkl', '').split('_')
            policy = None
            # Check if last part is a policy (not a number)
            if parts and not parts[-1].isdigit():
                policy = parts[-1]
            return cls(cls._convert_to_metric_data(data), policy=policy)
        
    @staticmethod
    def merge_results_as_dict(paths):
        """
        Merge multiple Results files into a dictionary of Results objects.
        
        Parameters
        ----------
        paths : list
            List of file paths to Results pickle files
            
        Returns
        -------
        dict
            Dictionary with structure { expr_name: Results_object }
        """
        merged_dict = {}
        
        for i, path in enumerate(paths):
            # Load the Results object from the path
            result_obj = Results.load(path)
            
            # Get the experiment name from the Results object
            expr_name = result_obj.expr[0]
            
            # Store it in the merged dictionary with the model name as key
            merged_dict[expr_name] = result_obj
        
        return merged_dict
       
    @classmethod
    def from_raw_dict(cls, results: Dict, policy: Optional[str] = None):
        return cls(cls._convert_to_metric_data(results), policy=policy)

    def get_stats(self, expr: str, rep: int, offset: int, prefix: int, suffix: int, metric: str) -> MetricData:
        return self.data[expr][rep][offset][prefix][suffix][metric]

    def get_all_metrics(self, expr: str, rep: int, offset: int, prefix: int, suffix: int) -> Dict[str, MetricData]:
        return self.data[expr][rep][offset][prefix][suffix]

    def get_dimensions(self) -> Dict[str, List]:
        return {
            'expr': self.expr,
            'repetitions': self.repetitions, 
            'offsets': self.offsets,
            'prefixes': self.prefixes,
            'suffixes': self.suffixes,
            'metrics': self.metrics
        }
    
    def save(self, folder='sparse'):
        base_path=f'/iopsstor/scratch/cscs/xyixuan/PDM/results/{folder}'

        offsets_str = '_'.join(map(str, self.offsets))
        prefixes_str = '_'.join(map(str, self.prefixes))
        suffixes_str = '_'.join(map(str, self.suffixes))
        
        if self.policy:
            name = f"offset_{offsets_str}_prefix_{prefixes_str}_suffix_{suffixes_str}_{self.policy}.pkl"
        else:
            name = f"offset_{offsets_str}_prefix_{prefixes_str}_suffix_{suffixes_str}.pkl"

        path = f"{base_path}/{self.expr[0]}/{name}"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Check if file exists and delete it
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted existing file: {path}")
        
        # Save the new file
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        
        print(f"Saved data to: {path}")

@dataclass
class EvalConfig:
    """Configuration for evaluation parameters
    
    All parameters (offset, prefix_length, suffix_length) are specified as lists.
    For non-varying parameters, simply provide a single-element list.
    
    Example:
        config = EvalConfig(
            base_path="path/to/data",
            expr="experiment1",
            repetitions=np.array([0, 1, 2]),
            policy="policy1",
            offsets=[100, 200, 300],        # varying parameter
            prefix_lengths=[500],            # non-varying parameter
            suffix_lengths=[100, 200, 300]   # varying parameter
        )
    """
    base_path: str
    expr: str
    repetitions: np.ndarray
    policy: str
    offsets: List[int]
    prefix_lengths: List[int]
    suffix_lengths: List[int]


def eval_expr(config: EvalConfig) -> Results:
    """
    Evaluate expressions with a consistent 5-level nested dictionary structure:
    expr -> repetition -> offset -> prefix_length -> suffix_length -> metrics

    Args:
        config: EvalConfig object containing evaluation parameters
        
    Returns:
        Dict with structure:
        {
            expr: {
                rep: {
                    offset: {
                        prefix_length: {
                            suffix_length: {
                                metric_name: {
                                    'scores': np.array(...),
                                    'mean': float,
                                    'std': float
                                }
                            }
                        }
                    }
                }
            }
        }
    """
    data = {config.expr: {}}

    for r in tqdm(config.repetitions, desc="Processing repetitions"):
        data[config.expr][r] = {}
        for offset in tqdm(config.offsets, desc="Processing offsets", leave=False):
            data[config.expr][r][offset] = {}
            for prefix_length in tqdm(config.prefix_lengths, leave=False):
                data[config.expr][r][offset][prefix_length] = {}
                for suffix_length in tqdm(config.suffix_lengths, leave=False):
                    metrics = calculate_metrics(config, r, offset, prefix_length, suffix_length)
                    data[config.expr][r][offset][prefix_length][suffix_length] = metrics

    return Results.from_raw_dict(data, policy=config.policy)


def calculate_metrics(config, rep, offset, prefix_length, suffix_length) -> Dict[str, Dict]:
    data = load_inference_data(
        f"{config.base_path}/{config.expr}/inference",
        rep=rep,
        policy=config.policy,
        offset=offset, 
        len_prefix=prefix_length,
        len_suffix=suffix_length
    )

     # First check if we already have the metrics computed during inference
    has_precomputed_metrics = all(metric in data.column_names 
                              for metric in ['TTR_ref', 'TTR_gen', 'Rouge-L'])

    # Only run the batch_rouge_ttr_calc if metrics were not precomputed
    if not has_precomputed_metrics:
        eval_results = data.map(
            batch_rouge_ttr_calc,
            batched=True,
            batch_size=5,
            num_proc=100,
            remove_columns=data.column_names,
            fn_kwargs={
                "true_key": "true_suffix", 
                "gen_key": "generated_suffix", 
                "len_suffix": suffix_length
            }
        )
    else:
        # Only keep the specific metrics we need
        needed_columns = ['lcs_norm', 'TTR_ref', 'TTR_gen', 'Rouge-L']
        available_columns = [col for col in needed_columns if col in data.column_names]
        eval_results = data.select_columns(available_columns)

    metrics = {
        'NLL': {
            'scores': data['nll_mean'],
            'mean': np.mean(data['nll_mean']),
            'std': np.std(data['nll_mean'])
        },
        'PPL': {
            'scores': data['perplexity'],
            'mean': np.mean(data['perplexity']),
            'std': np.std(data['perplexity'])
        },
        'Ref_NLL': {
            'scores': data['ref_nll_mean'],
            'mean': np.mean(data['ref_nll_mean']),
            'std': np.std(data['ref_nll_mean'])
        },
        'Ref_PPL': {
            'scores': data['ref_perplexity'],
            'mean': np.mean(data['ref_perplexity']),
            'std': np.std(data['ref_perplexity'])
        },
        'Rouge-L': {
            'scores': data['Rouge-L'],
            'mean': np.mean(data['Rouge-L']),
            'std': np.std(data['Rouge-L'])
        },
        'LCS': {
            'scores': data['lcs_norm'],
            'mean': np.mean(data['lcs_norm']),
            'std': np.std(data['lcs_norm'])
        },
        'TTR_ref': {
            'scores': data['TTR_ref'],
            'mean': np.mean(data['TTR_ref']),
            'std': np.std(data['TTR_ref'])
        },
        'TTR_gen': {
            'scores': data['TTR_gen'],
            'mean': np.mean(data['TTR_gen']),
            'std': np.std(data['TTR_gen'])
        },
    }

    # Calculate exact match statistics
    lcs_norm_values = np.array(data['lcs_norm'])
    exact_match = sum(lcs_norm_values == 1.0)
    match_75 = sum(lcs_norm_values >= 0.75)
    match_50 = sum(lcs_norm_values >= 0.50)
    match_25 = sum(lcs_norm_values >= 0.25)
    
    # Calculate percentages
    total_samples = len(lcs_norm_values)
    exact_match_pct = (exact_match / total_samples) if total_samples > 0 else 0
    match_75_pct = (match_75 / total_samples) if total_samples > 0 else 0
    match_50_pct = (match_50 / total_samples) if total_samples > 0 else 0
    match_25_pct = (match_25 / total_samples) if total_samples > 0 else 0
    
    # Add to metrics dictionary
    metrics['exact_match'] = {
        'scores': exact_match,
        'mean': exact_match_pct,
        'std': 0
    }
    metrics['match_75'] = {
        'scores': match_75,
        'mean': match_75_pct, 
        'std': 0
    }
    metrics['match_50'] = {
        'scores': match_50,
        'mean': match_50_pct,
        'std': 0
    }
    metrics['match_25'] = {
        'scores': match_25,
        'mean': match_25_pct,
        'std': 0
    }

    return metrics


def get_repetitions(inference_path):
    """Get repetition values from inference directories."""
    return sorted(
        int(path.name.split('_')[1]) 
        for path in inference_path.glob("rep_*_greedy")
        if path.name.split('_')[1].isdigit()
    )


def get_repetition_mean_df(
   results: Results, 
   metric: str,
   offset: Optional[int] = None,
   prefix: Optional[int] = None,
   suffix: Optional[int] = None
) -> pd.DataFrame:
    if sum(x is not None for x in [offset, prefix, suffix]) != 2:
        raise ValueError("Exactly two parameters must be provided")
        
    # Determine which parameter varies
    if offset is None:
        vary_by, iterations = 'offset', results.offsets
    elif prefix is None:
        vary_by, iterations = 'prefix', results.prefixes
    else:
        vary_by, iterations = 'suffix', results.suffixes

    # Store the fixed values
    fixed_vals = {
        'offset': offset,
        'prefix': prefix,
        'suffix': suffix
    }

    # Create data dictionary with correct value selection
    data = {}
    for rep in results.repetitions:
        data[rep] = {}
        for val in iterations:
            # Select the correct values based on which parameter varies
            current_offset = val if vary_by == 'offset' else fixed_vals['offset']
            current_prefix = val if vary_by == 'prefix' else fixed_vals['prefix']
            current_suffix = val if vary_by == 'suffix' else fixed_vals['suffix']
            
            # Get stats
            stats = results.get_stats(
                results.expr[0],
                rep,
                current_offset,
                current_prefix,
                current_suffix,
                metric
            )
            data[rep][val] = stats.mean

    df = pd.DataFrame(data).T
    df.index.name = 'repetition'
    df.columns.name = vary_by[:-1] if vary_by.endswith('s') else vary_by  # Remove 's' if present

    return df


def print_repetition_stats(
    results: Results,
    metric: Optional[str] = None,
    offset: Optional[int] = None,
    prefix: Optional[int] = None,
    suffix: Optional[int] = None
):
    if sum(x is not None for x in [offset, prefix, suffix]) != 2:
        raise ValueError("Exactly two parameters must be provided")
    
    # Determine which parameter varies
    if offset is None:
        vary_by, iterations = 'offset', results.offsets
    elif prefix is None:
        vary_by, iterations = 'prefix', results.prefixes
    else:
        vary_by, iterations = 'suffix', results.suffixes

    # Store the fixed values
    fixed_vals = {
        'offset': offset,
        'prefix': prefix,
        'suffix': suffix
    }

    for rep in results.repetitions:
        print(f"\n=== Summary for Repetition {rep:3d} ===\n")
        for val in iterations:
            # Select the correct values based on which parameter varies
            current_offset = val if vary_by == 'offset' else fixed_vals['offset']
            current_prefix = val if vary_by == 'prefix' else fixed_vals['prefix']
            current_suffix = val if vary_by == 'suffix' else fixed_vals['suffix']
            
            print(f"{vary_by.capitalize()}: {val:3d}")
            metrics_to_show = [metric] if metric else results.metrics
            
            for m in metrics_to_show:
                stats = results.get_stats(
                    results.expr[0],
                    rep,
                    current_offset,
                    current_prefix,
                    current_suffix,
                    m
                )
                print(f" {m:9s} | Mean = {stats.mean:.3f}, Std = {stats.std:.3f}")
            print()


def plot_repetition_metric_dists(results: Results, metric: str, offset: int = 0, prefix: int = 500):
    n_reps = len(results.repetitions)
    n_suffix = len(results.suffixes)

    fig, axes = plt.subplots(n_suffix, n_reps, figsize=(5*n_reps, 4*n_suffix))
    colors = plt.cm.tab20(np.linspace(0, 1, n_suffix))

    axes = np.array([[axes]]) if n_suffix == 1 and n_reps == 1 else \
            axes.reshape(1, -1) if n_suffix == 1 else \
            axes.reshape(-1, 1) if n_reps == 1 else axes

    for suffix_idx, suffix in enumerate(results.suffixes):
        for rep_idx, rep in enumerate(results.repetitions):
            ax = axes[suffix_idx, rep_idx]
            
            if metric == 'TTR':
                gen_scores = results.get_stats(results.expr[0], rep, offset, prefix, suffix, 'TTR_gen').scores
                ref_scores = results.get_stats(results.expr[0], rep, offset, prefix, suffix, 'TTR_ref').scores
                
                ax.hist(gen_scores, bins=20, color=colors[suffix_idx], 
                        edgecolor='black', linewidth=1.2, alpha=0.7, label='Generated')
                ax.hist(ref_scores, bins=20, color='gray', 
                        edgecolor='black', linewidth=1.2, alpha=0.5, label='Reference')
                ax.legend(fontsize=12, facecolor='white', edgecolor='black', framealpha=1)
            else:
                scores = results.get_stats(results.expr[0], rep, offset, prefix, suffix, metric).scores
                ax.hist(scores, bins=20, color=colors[suffix_idx], edgecolor='black', linewidth=1.2)
            
            ax.set_xlabel(f'{metric} Score')
            ax.set_ylabel('Frequency' if rep_idx == 0 else '')
            if metric == 'Rouge-L':
                ax.set_xlim(0, 1)
            if suffix_idx < n_suffix - 1:
                ax.set_xlabel('')

    for suffix_idx, suffix in enumerate(results.suffixes):
        axes[suffix_idx, 0].text(-0.3, 0.5, f'Suffix {suffix}', fontsize=20, rotation=90,
                                transform=axes[suffix_idx, 0].transAxes, verticalalignment='center')

    for rep_idx, rep in enumerate(results.repetitions):
        axes[0, rep_idx].text(0.5, 1.2, f'Rep {rep}', fontsize=20,
                            transform=axes[0, rep_idx].transAxes, horizontalalignment='center')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.01, f'{metric} Score Distribution',
            horizontalalignment='center', verticalalignment='bottom', fontsize=30)
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
        e.g., ['greedy', 'beam_nb5', 'nucleus_p0.9']
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
    if sum(x is not None for x in [offset, prefix, suffix]) != 2:
        raise ValueError("Exactly two parameters must be provided")
    
    # Construct file paths and load results
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
    
    # Always use multi-metric display format (even for single metric)
    print(f"\n=== Metric Comparison for Repetition {repetition} ===")
    print(f"Fixed: prefix={prefix}, suffix={suffix}" if offset is None else
          f"Fixed: offset={offset}, suffix={suffix}" if prefix is None else
          f"Fixed: offset={offset}, prefix={prefix}")
    print()
    
    # Define column widths based on show_std
    if show_std:
        value_width = 15  # "  0.965 ± 0.135"
        policy_col_width = 15
    else:
        value_width = 9   # "  0.965"
        policy_col_width = 9
    
    # Build header rows
    header1_parts = [f"{' ':>8} |"]
    header2_parts = [f"{vary_by.capitalize():>8} |"]
    
    for metric in metrics_list:
        # Calculate metric header span width
        # For each metric group: policies are separated by " | "
        metric_group_width = len(policies) * policy_col_width + (len(policies) - 1) * 3
        header1_parts.append(f" {metric:^{metric_group_width}} |")
        
        # Add policy names
        for i, policy in enumerate(policies):
            if i == 0:
                header2_parts.append(f" {policy:>{policy_col_width}} ")
            else:
                header2_parts.append(f"| {policy:>{policy_col_width}} ")
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
            for i, (policy, results) in enumerate(results_dict.items()):
                try:
                    stats = results.get_stats(
                        results.expr[0],
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
                        row_parts.append(f" {value:>{policy_col_width}} ")
                    else:
                        row_parts.append(f"| {value:>{policy_col_width}} ")
                except:
                    if i == 0:
                        row_parts.append(f" {'N/A':>{policy_col_width}} ")
                    else:
                        row_parts.append(f"| {'N/A':>{policy_col_width}} ")
            row_parts.append("|")
        
        print("".join(row_parts))