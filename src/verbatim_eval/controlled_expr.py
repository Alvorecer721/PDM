from typing import List, Dict, Any, Literal, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
import os

from .rouge_ttr import batch_rouge_ttr_calc
from .utils import load_inference_data

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
            return cls(cls._convert_to_metric_data(pickle.load(f)))
       
    @classmethod
    def from_raw_dict(cls, results: Dict):
        return cls(cls._convert_to_metric_data(results))

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
    
    def save(self):
        base_path='/capstor/users/cscs/xyixuan/PDM/results/sparse'

        offsets_str = '_'.join(map(str, self.offsets))
        prefixes_str = '_'.join(map(str, self.prefixes))
        suffixes_str = '_'.join(map(str, self.suffixes))
        name = f"offset_{offsets_str}_prefix_{prefixes_str}_suffix_{suffixes_str}.pkl"

        path = f"{base_path}/{self.expr[0]}/{name}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)


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

    return Results.from_raw_dict(data)


def calculate_metrics(config, rep, offset, prefix_length, suffix_length) -> Dict[str, Dict]:
    data = load_inference_data(
        f"{config.base_path}/{config.expr}/inference",
        rep=rep,
        policy=config.policy,
        offset=offset, 
        len_prefix=prefix_length,
        len_suffix=suffix_length
    )

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

    metrics = {
        'NLL': {
            'scores': data['nll_mean'],
            'mean': np.mean(data['nll_mean']),
            'std': np.std(data['nll_mean'])
        }
    }

    for metric in eval_results.column_names:
        scores = np.array(eval_results[metric])
        metrics[metric] = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }

    return metrics


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