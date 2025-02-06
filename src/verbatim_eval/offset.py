from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .rouge_ttr import batch_rouge_ttr_calc
from .utils import load_inference_data

@dataclass
class EvalOffsetConfig:
    """Configuration for evaluation parameters"""
    base_path: str
    expr: str
    repetitions: np.ndarray
    policy: str
    offsets: List[int]
    len_prefix: str
    len_suffix: str

def eval_expr_by_offset(config: EvalOffsetConfig) -> Dict:
    """
    Evaluate expressions by offset with improved error handling
    """
    results = {}
    results[config.expr] = {}

    pbar1 = tqdm(config.repetitions, desc="Processing repetitions")
    
    for r in pbar1:
        pbar1.set_description(f"Processing repetition {r}")
        results[config.expr][r] = {}

        pbar2 = tqdm(config.offsets, desc="Processing offsets", leave=False)
        for offset in pbar2:
            pbar2.set_description(f"Processing offset {offset}")
            
            try:
                data_path = f"{config.base_path}/{config.expr}/inference"
                data = load_inference_data(
                    data_path, 
                    rep=r,
                    policy=config.policy,
                    offset=offset,
                    len_prefix=config.len_prefix,
                    len_suffix=config.len_suffix
                )

                # Calculate metrics
                eval_results = data.map(
                    batch_rouge_ttr_calc,
                    batched=True, 
                    batch_size=5,
                    num_proc=100,
                    desc=f"Calculating metrics for rep={r}, offset={offset}",
                    remove_columns=data.column_names,
                    fn_kwargs={
                        "true_key": "true_suffix",
                        "gen_key": "generated_suffix",
                        "len_suffix": config.len_suffix,
                    }
                )

                results[config.expr][r][offset] = {}

                # NLL
                results[config.expr][r][offset]['NLL'] = {
                    'scores': data['nll_mean'],
                    'mean': np.mean(data['nll_mean']),
                    'std': np.std(data['nll_mean'])
                }

                # Store other metrics
                for metric in eval_results.column_names:
                    scores = np.array(eval_results[metric])
                    results[config.expr][r][offset][metric] = {
                        'scores': scores,
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    }
            except Exception as e:
                print(f"\nError processing rep={r}, offset={offset}: {str(e)}")
                continue

        pbar2.close()
    
    pbar1.close()
    return results


def format_offset_metric_df(result, metric):
    """
    Convert experiment results into a pandas DataFrame.
    Args:
        result: Dictionary containing a single experiment's results
        metric: String specifying which metric to show
    Returns:
        DataFrame with offsets as rows and (repetition, stat) as column MultiIndex
    """
    # Get the only experiment's data
    rep_dict = next(iter(result.values()))
    
    # Get all unique offsets and repetitions
    offsets = sorted(set(off for rep_d in rep_dict.values() for off in rep_d.keys()))
    repetitions = sorted(rep_dict.keys())
    
    # Create MultiIndex for columns
    columns = pd.MultiIndex.from_product([
        repetitions,
        ['mean', 'std']
    ], names=['repetition', 'stat'])
    
    # Initialize DataFrame with NaN values
    df = pd.DataFrame(index=offsets, columns=columns)
    df.index.name = 'offset'
    
    # Fill the DataFrame
    for rep, offset_dict in rep_dict.items():
        for offset, metrics_dict in offset_dict.items():
            if metric in metrics_dict:
                df.loc[offset, (rep, 'mean')] = metrics_dict[metric]['mean']
                df.loc[offset, (rep, 'std')] = metrics_dict[metric]['std']
    
    return df


def print_offset_metrics(results: Dict, metric=None):
    """
    Log metrics summary for experiment with repetitions and offset locations.
    Args:
        results: Dictionary output from eval_expr_by_offset
        metric: Optional string to filter for a specific metric
    """
    for expr, rep_dict in results.items():
        print(f"\n=== Summary for Experiment: {expr} ===")
        for rep, offset_dict in rep_dict.items():
            print(f"\n === Summary for Repetition {rep:3d} ===")
            for offset, metrics_dict in offset_dict.items():
                print(f" Offset {offset:3d}")
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


if __name__ == "__main__":
    config = EvalOffsetConfig(
        base_path="/iopsstor/scratch/cscs/xyixuan/experiment",
        expr="llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_10200350",
        repetitions=np.array([1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128]),
        policy="greedy",
        offsets=[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100],
        len_prefix=500,
        len_suffix=500
    )

    results = eval_expr_by_offset(config)
    print(results)