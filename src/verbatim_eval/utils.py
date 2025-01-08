from pathlib import Path
from datasets import load_dataset
import numpy as np

def load_inference_data(base_dir, step=None, consumed=None, rep=None):
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

    if rep is not None:
        file_path = Path(base_dir) / f"rep_{rep}"
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

