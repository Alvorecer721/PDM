"""
Calculate contamination from inference outputs to swapped parts using DataTrove.

This script specifically measures how much of the model's generated text (inference suffixes)
matches the swapped parts of the Gutenberg dataset. This helps understand if the model
is memorizing and reproducing the swapped content.

Contamination Direction: Inference → Swapped Parts

Usage:
------
python src/decont/calc_suffix_to_swapped_conta.py \
      --swapped-indices-dir /iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/swapped_part \
      --inference-indices-dir /iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/suffix/8b/offset_0_prefix_500_suffix_500 \
      --output-file /iopsstor/scratch/cscs/xyixuan/PDM/results/decont/contamination_results_suffix_swapped.json \
      --num-proc 9

Output:
-------
JSON file with contamination statistics per repetition:
{
    "rep_1": {"contamination_ratio": 0.023, "matching_ngrams": 45, "total_ngrams": 2000},
    "rep_2": {"contamination_ratio": 0.045, "matching_ngrams": 90, "total_ngrams": 2000},
    ...
}
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.decont.commons import load_hash_index, calculate_contamination_ratio, find_repetition_based_matches, print_contamination_summary

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Calculate contamination from inference outputs to swapped parts')
    
    parser.add_argument(
        '--swapped-indices-dir',
        type=str,
        required=True,
        help='Directory containing swapped part indices (e.g., /path/to/indices/swapped_part)'
    )
    
    parser.add_argument(
        '--inference-indices-dir',
        type=str,
        required=True,
        help='Directory containing inference indices (e.g., /path/to/indices/suffix/8b/offset_0_prefix_500_suffix_500)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='inference_to_swapped_contamination.json',
        help='Output file for contamination results'
    )
    
    parser.add_argument(
        '--num-proc',
        type=int,
        default=None,
        help='Number of parallel processes (default: all CPUs)'
    )
    
    parser.add_argument(
        '--n-gram-size',
        type=int,
        default=13,
        help='N-gram size (should match index building)'
    )
    
    return parser


def calculate_contamination_single_rep(args_tuple):
    """Calculate contamination for a single repetition."""
    rep_num, swapped_index_path, inference_index_path = args_tuple
    
    try:
        # Check if both index files exist
        if not swapped_index_path.exists():
            return {
                'rep': rep_num,
                'error': f'Swapped index not found: {swapped_index_path}'
            }
        
        if not inference_index_path.exists():
            return {
                'rep': rep_num,
                'error': f'Inference index not found: {inference_index_path}'
            }
        
        # Load both hash indices
        swapped_hashes = load_hash_index(swapped_index_path)
        inference_hashes = load_hash_index(inference_index_path)
        
        # Calculate contamination using common function
        # Note: inference is source, swapped is target (checking if inference contains swapped)
        stats = calculate_contamination_ratio(inference_hashes, swapped_hashes)
        
        return {
            'rep': rep_num,
            'contamination_ratio': stats['contamination_ratio'],
            'matching_ngrams': stats['matching_ngrams'],
            'total_inference_ngrams': stats['total_source_ngrams'],
            'total_swapped_ngrams': stats['total_target_ngrams'],
            'success': True
        }
        
    except Exception as e:
        return {
            'rep': rep_num,
            'error': str(e)
        }

def analyze_contamination_parallel(args):
    """Main function to analyze contamination across all repetitions."""
    swapped_dir = Path(args.swapped_indices_dir)
    inference_dir = Path(args.inference_indices_dir)
    
    # Use common function to find matching repetitions
    swapped_reps, inference_reps, common_reps = find_repetition_based_matches(
        swapped_dir, 
        inference_dir,
        source_pattern="rep_*_text_*",
        target_pattern="rep_*_greedy_text"
    )
    
    print(f"\nCalculating Inference → Swapped Contamination")
    print(f"="*50)
    print(f"Found indices for repetitions: {common_reps}")
    print(f"Swapped indices: {len(swapped_reps)} reps")
    print(f"Inference indices: {len(inference_reps)} reps")
    print(f"Common reps to analyze: {len(common_reps)}")
    
    if not common_reps:
        print("No common repetitions found between swapped and inference indices!")
        return {}
    
    # Prepare arguments for parallel processing
    process_args = [
        (rep, swapped_reps[rep], inference_reps[rep])
        for rep in common_reps
    ]
    
    # Determine number of workers
    num_workers = min(args.num_proc or cpu_count(), len(common_reps))
    print(f"\nUsing {num_workers} parallel workers for contamination analysis")
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(calculate_contamination_single_rep, process_args),
            total=len(common_reps),
            desc="Calculating contamination"
        ))
    
    # Organize results by repetition
    contamination_results = {}
    errors = []
    
    for result in results:
        rep = result['rep']
        if 'error' in result:
            errors.append(f"Rep {rep}: {result['error']}")
        else:
            contamination_results[f'rep_{rep}'] = {
                'contamination_ratio': result['contamination_ratio'],
                'matching_ngrams': result['matching_ngrams'],
                'total_inference_ngrams': result['total_inference_ngrams'],
                'total_swapped_ngrams': result['total_swapped_ngrams']
            }
    
    # Report errors if any
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    return contamination_results


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print(f"\nInference → Swapped Contamination Analysis")
    print(f"="*50)
    print(f"Swapped indices directory: {args.swapped_indices_dir}")
    print(f"Inference indices directory: {args.inference_indices_dir}")
    print(f"N-gram size: {args.n_gram_size}")
    
    # Run contamination analysis
    results = analyze_contamination_parallel(args)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary using common function
    # Convert rep_N keys to just N for cleaner display
    display_results = {}
    for key, value in results.items():
        rep_num = int(key.split('_')[1])
        display_results[str(rep_num)] = value
    
    print_contamination_summary(display_results, source_name="Inference", target_name="Swapped")
    
    # Create a CSV for easy analysis
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("repetition,contamination_ratio,matching_ngrams,total_inference_ngrams,total_swapped_ngrams\n")
        for rep_key, data in sorted(results.items(), key=lambda x: int(x[0].split('_')[1])):
            rep_num = int(rep_key.split('_')[1])
            f.write(f"{rep_num},{data['contamination_ratio']},{data['matching_ngrams']},"
                   f"{data['total_inference_ngrams']},{data['total_swapped_ngrams']}\n")
    
    print(f"CSV results saved to: {csv_path}")

if __name__ == "__main__":
    main()