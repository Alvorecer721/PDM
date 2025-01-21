from ignite.metrics import RougeL
from typing import List
import numpy as np
from .utils import load_inference_data
from tqdm import tqdm


def batch_rouge_ttr_calc(batch, true_key, gen_key, len_suffix):
    """
    Calculate ROUGE-L scores for a batch of true and generated sequences.

    Args:
        batch (dict): Batch of data containing true and generated sequences.
        true_key (str, optional): Key for true sequences. Defaults to "true_suffix".
        gen_key (str, optional): Key for generated sequences. Defaults to "generated_suffix".

    Returns:
        dict: Dictionary containing arrays of ROUGE-L scores and summary statistics.
    """


    rouge_scores = []
    ttr_ref_scores = []
    ttr_gen_scores = []
    rouge_metric = RougeL(multiref="best")

    for true_seq, gen_seq in zip(batch[true_key], batch[gen_key]):

        assert len_suffix <= len(true_seq), "Length of suffix is greater than sequence length."

        # Wrap sequences as required by the metric
        ref_slice = true_seq[:len_suffix]
        gen_slice  = gen_seq[:len_suffix]

        ttr_ref = len(set(ref_slice)) / len(ref_slice)
        ttr_gen  = len(set(gen_slice)) / len(gen_slice)

        # Reset metric for new pair
        rouge_metric.reset()
        rouge_metric.update(([ref_slice], [[gen_slice]]))
        rouge = rouge_metric.compute()
        
        # Extract F1 score
        rouge_scores.append(rouge['Rouge-L-F'])
        ttr_ref_scores.append(ttr_ref)
        ttr_gen_scores.append(ttr_gen)

    return {
        "Rouge-L": rouge_scores,
        "TTR_ref": ttr_ref_scores,
        "TTR_gen": ttr_gen_scores
    }
            

def eval_rouge_ttr(base_path: str, experiments: List[str], repetitions: np.ndarray, len_suffix: int, policy: str):
    """
    Evaluate the following metrics for a given experiment and repetitions.
    1. ROUGE-L
    2. TTR (Type-Token Ratio): Ratio of unique tokens to total tokens in a sequence.

    Args:
        expr (str): Name of the experiment.
        repetitions (np.ndarray): Array of repetition numbers.

    Returns:
        dict: Dictionary containing arrays of ROUGE-L scores and summary statistics.
    """
    results = {}
    
    # Load inference data
    for expr in experiments:
        results[expr] = {}

        pbar = tqdm(repetitions, desc="Processing repetition set")
        for r in pbar:
            pbar.set_description(f"Processing repetition set {r}")  
            data_path = f"{base_path}/{expr}/inference"
            data = load_inference_data(data_path, rep=r, policy=policy)

            # Calculate ROUGE-L scores
            eval_results = data.map(
                batch_rouge_ttr_calc, 
                batched=True, 
                batch_size=10, 
                num_proc=8, 
                desc=f"Calculating metrics for rep={r}", 
                remove_columns=data.column_names,
                fn_kwargs={
                    "true_key": "true_suffix",
                    "gen_key": "generated_suffix",
                    "len_suffix": len_suffix, 
                })
            
            results[expr][r] = {}

            # NLL
            results[expr][r]['NLL'] = {
                'scores': data['nll_mean'],
                'mean': np.mean(data['nll_mean']),
                'std': np.std(data['nll_mean'])
            }

            # Store results in dictionary
            for metric in eval_results.column_names:
                
                scores = np.array(eval_results[metric])

                results[expr][r][metric] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }

    return results


def log_metrics(results_by_experiment):
    """
    Log summary of *all* metrics under each experiment and repetition.
    """
    for experiment, rep_dict in results_by_experiment.items():
        print(f"\n=== Summary for Experiment: {experiment} ===")
        for repetition, metrics_dict in rep_dict.items():
            print(f"  Repetition {repetition:3d}")
            # metrics_dict is e.g. {"Rouge-L": {...}, "TTR_ref": {...}, ...}
            for metric_name, stats in metrics_dict.items():
                print(
                    f"    {metric_name:<10} | "
                    f"Mean = {stats['mean']:.3f}, "
                    f"Std = {stats['std']:.3f}"
                )


def log_metric(results, metric="Rouge-L"):
    """
    For each repetition, print the selected metric for each experiment.

    Parameters
    ----------
    results : dict
        Nested dict of form:
            {
                experiment_name: {
                    repetition_index: {
                        metric_name: {
                            'scores': np.ndarray,
                            'mean': float,
                            'std': float
                        },
                        ...
                    },
                    ...
                },
                ...
            }
    metric : str
        A single metric name to log (e.g., "Rouge-L").
    """
    # Collect and sort all repetition indices across experiments
    all_reps = sorted({rep for exp_data in results.values() for rep in exp_data})

    # Determine a suitable width for experiment name alignment:
    # e.g., find the longest experiment name and add some padding.
    max_exp_length = max(len(exp_name) for exp_name in results.keys())
    field_width = max(20, max_exp_length + 2)  # minimum 20 or expand if longer names


    # Main loop: repetition first, then each experiment
    for rep in all_reps:
        print(f"\n=== Repetition {rep} {metric} ===")
        for experiment in sorted(results.keys()):

            # Extract the metric data (e.g., {"mean": 0.7, "std": 0.05, "scores": ...})
            data = results[experiment][rep].get(metric)

            print(
                f"  {experiment:<{field_width}}: Mean = {data['mean']:.3f} | Std = {data['std']:.3f}"
            )
