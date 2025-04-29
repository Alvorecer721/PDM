import os
import json
import numpy as np
from pathlib import Path
from evaluate import load
from utils import load_inference_data
from transformers import AutoTokenizer
import argparse


def compute_and_save_mauve_scores(data, tokenizer, expr_name, rep, offset, prefix_length, suffix_length, sample_size=None):
    """
    Compute MAUVE scores and save them directly to a JSON file.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing 'true_suffix' and 'generated_suffix' as token IDs
        This should be loaded using your load_inference_data function
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to decode token IDs to text
    expr_name : str
        Experiment name
    rep : int
        Repetition number
    offset : int
        Offset value
    prefix_length : int
        Prefix length
    suffix_length : int
        Suffix length
    sample_size : int, optional
        If provided, only use this many samples for computation
    """
    
    # Set environment variables to avoid OpenBLAS warnings
    os.environ['OPENBLAS_NUM_THREADS'] = '128'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    
    # Create output directory with the specified structure using pathlib
    output_dir = Path('/capstor/users/cscs/xyixuan/PDM/results/mauve') / expr_name / f"offset_{offset}" / f"prefix_{prefix_length}_suffix_{suffix_length}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the data as is
    if sample_size is not None and sample_size < len(data['true_suffix']):
        # Use only a subset if sample_size is specified
        true_suffix = data['true_suffix'][:sample_size]
        generated_suffix = data['generated_suffix'][:sample_size]
    else:
        true_suffix = data['true_suffix']
        generated_suffix = data['generated_suffix']
    
    print(f"Decoding {len(true_suffix)} samples...")
    true_texts = [tokenizer.decode(ids) for ids in true_suffix]
    generated_texts = [tokenizer.decode(ids) for ids in generated_suffix]
    
    print(f"Computing MAUVE scores...")
    
    try:
        # Use a smaller batch size and limit sequence length for efficiency
        mauve = load('mauve')
        mauve_results = mauve.compute(
            predictions=generated_texts,
            references=true_texts,
            device_id=0,
            verbose=True
        )
        
        # Extract the MAUVE score and round to 3 decimal places
        mauve_score = round(float(mauve_results.mauve), 3)
        
        # Create a simple results dictionary for JSON
        results_dict = {
            "mauve_score": mauve_score,
        }
        
        # Save the results to JSON using the specified path
        output_file = output_dir / f"rep_{rep}.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Saved MAUVE scores to: {output_file}")
        print(f"MAUVE score: {mauve_score}")
        
        return mauve_score
    
    except Exception as e:
        print(f"Error computing MAUVE score: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mauve Computation")
    parser.add_argument("--expr", type=str, required=True, help="Experiment name")
    parser.add_argument("--base-path", type=str, 
                        default="/iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/",
                        help="Base path for experiments")
    parser.add_argument("--offsets", type=int, nargs="+", 
                        default=[0],
                        help="List of offsets")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", 
                        default=[500],
                        help="List of prefix lengths")
    parser.add_argument("--suffix-lengths", type=int, nargs="+", 
                        default=[500],
                        help="List of suffix lengths")
    parser.add_argument("--repetitions", type=int, nargs="+", 
                        default=[0, 1, 2, 4, 8, 16, 32, 64, 128],
                        help="List of repetition numbers")
    parser.add_argument("--policy", type=str, default="greedy", help="Generation policy")
    parser.add_argument("--mode", type=str, default="SparseGutenberg", choices=["SparseGutenberg", "DenseGutenberg", "Goldfish-H-Ablation", "Offset-Effect"], help="Mode for saving results")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for MAUVE computation")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.model_max_length = 200_000
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get full experiment path using pathlib
    base_path = Path(args.base_path)
    expr_path = Path(args.mode) / args.expr
    inference_dir = base_path / expr_path / "inference"
    
    # Loop through all parameter combinations
    for offset in args.offsets:
        for prefix_length in args.prefix_lengths:
            for suffix_length in args.suffix_lengths:
                for rep in args.repetitions:
                    print(f"\n=== Processing: offset={offset}, prefix={prefix_length}, suffix={suffix_length}, rep={rep} ===\n")
                
                    # Load data for this configuration
                    data = load_inference_data(
                        base_dir=str(inference_dir),  # Convert Path to string for compatibility
                        offset=offset,
                        len_prefix=prefix_length,
                        len_suffix=suffix_length,
                        rep=rep,
                        policy=args.policy,
                    )
                    
                    # Compute and save MAUVE scores
                    compute_and_save_mauve_scores(
                        data=data,
                        tokenizer=tokenizer,
                        expr_name=args.expr,
                        rep=rep,
                        offset=offset,
                        prefix_length=prefix_length,
                        suffix_length=suffix_length,
                        sample_size=args.sample_size
                    )