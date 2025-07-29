"""
Process Gutenberg swapped data and inference suffixes for decontamination analysis.

This script performs three main tasks:
1. Truncates swapped Gutenberg token data to first N tokens (swapped part) and decodes to text
2. Extracts and decodes generated suffixes from inference results
3. Builds n-gram indices for contamination analysis between the two datasets

Key Features:
- Parallel processing with HuggingFace's batched operations
- Efficient batch decoding for faster text conversion
- Architecture-specific caching to avoid conflicts
- Simplified output format optimized for decontamination (text only)

Usage Examples:
--------------
# Process both swapped and suffix data, then build indices:
python src/decont/build_suffix_vs_swap_hashes.py \
    --input-swapped-dir /iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_swapped \
    --input-inference-dir /iopsstor/scratch/cscs/xyixuan/Megatron-LM/logs/Meg-Runs/Offset-Effect/llama3-8b-15n-8192sl-60gbsz-swaps/inference/offset_0_prefix_500_suffix_500 \
    --output-base-dir /iopsstor/scratch/cscs/xyixuan/dataset/rebuttal \
    --process-swapped \
    --process-suffix \
    --build-indices \
    --num-proc 50 \
    --batch-size 500

# Process only swapped data with custom truncation:
python src/decont/build_suffix_vs_swap_hashes.py \
    --input-swapped-dir /path/to/swapped \
    --input-inference-dir /path/to/inference \
    --output-base-dir /path/to/output \
    --process-swapped \
    --truncate-length 4000 \
    --num-proc 4

# Process specific repetitions only:
python src/decont/build_suffix_vs_swap_hashes.py \
    --input-swapped-dir /path/to/swapped \
    --input-inference-dir /path/to/inference \
    --output-base-dir /path/to/output \
    --process-swapped \
    --process-suffix \
    --build-indices \
    --repetitions 1,2,4,8,16,32,64,128

# Process only suffixes:
python src/decont/build_suffix_vs_swap_hashes.py \
    --input-swapped-dir /path/to/swapped \
    --input-inference-dir /path/to/inference \
    --output-base-dir /path/to/output \
    --process-suffix \
    --model-size 8b

# Build indices only (assumes text files already exist):
python src/decont/build_suffix_vs_swap_hashes.py \
    --input-swapped-dir /path/to/swapped \
    --input-inference-dir /path/to/inference \
    --output-base-dir /path/to/output \
    --build-indices \
    --n-gram-size 13

# Run swapped and suffix processing in parallel (two terminals):
# Terminal 1:
python src/decont/build_suffix_vs_swap_hashes.py ... --process-swapped --build-indices --num-proc 25
# Terminal 2:
python src/decont/build_suffix_vs_swap_hashes.py ... --process-suffix --build-indices --num-proc 25

Performance Notes:
- Text processing uses parallel workers (--num-proc) for speed
- Index building uses single worker (DataTrove requirement)
- Default batch size is 1000, increase for better GPU utilization
- If --num-proc is not specified, uses all available CPUs

Output Structure:
----------------
output-base-dir/
├── swapped_part/
│   ├── rep_1_text_4000.jsonl
│   ├── rep_2_text_4000.jsonl
│   └── ...
└── suffix/
    └── 8b/
        └── offset_0_prefix_500_suffix_500/
            ├── rep_1_greedy_text.jsonl
            ├── rep_2_greedy_text.jsonl
            └── ...

indices-output-dir/
└── indices/
    ├── swapped_part/
    │   ├── rep_1_text_4000/
    │   │   ├── input.index.hashes
    │   │   └── stats.json
    │   └── ...
    └── suffix/
        └── 8b/
            └── offset_0_prefix_500_suffix_500/
                ├── rep_1_greedy_text/
                │   ├── input.index.hashes
                │   └── stats.json
                └── ...
"""

import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import platform
import argparse
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.verbatim_eval.utils import load_inference_data
from src.decont.commons import build_index_for_file, build_index_wrapper

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Process Gutenberg swapped data and inference suffixes for decontamination analysis')
    
    parser.add_argument(
        '--truncate-length',
        type=int,
        default=4000,
        help='Number of tokens to keep from swapped data (default: 4000)'
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        default='8b',
        help='Model size for inference suffix processing (default: 8b)'
    )
    
    parser.add_argument(
        '--input-swapped-dir',
        type=str,
        required=True,
        help='Input directory for swapped data'
    )
    
    parser.add_argument(
        '--input-inference-dir',
        type=str,
        required=True,
        help='Input directory for inference results (e.g., path/to/offset_0_prefix_500_suffix_500)'
    )
    
    parser.add_argument(
        '--output-base-dir',
        type=str,
        required=True,
        help='Base output directory'
    )
    
    parser.add_argument(
        '--process-swapped',
        action='store_true',
        help='Process swapped Gutenberg data'
    )
    
    parser.add_argument(
        '--process-suffix',
        action='store_true',
        help='Process generated suffixes'
    )
    
    parser.add_argument(
        '--tokenizer-model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Tokenizer model to use'
    )
    
    parser.add_argument(
        '--build-indices',
        action='store_true',
        help='Build n-gram indices after processing'
    )
    
    parser.add_argument(
        '--n-gram-size',
        type=int,
        default=13,
        help='Size of n-grams for indexing (default: 13)'
    )
    
    parser.add_argument(
        '--indices-output-dir',
        type=str,
        default='/iopsstor/scratch/cscs/xyixuan/PDM/results/decont',
        help='Output directory for n-gram indices'
    )
    
    parser.add_argument(
        '--num-proc',
        type=int,
        default=None,
        help='Number of processes for parallel processing (default: use all CPUs)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files and indices'
    )
    
    parser.add_argument(
        '--repetitions',
        type=str,
        default=None,
        help='Specific repetitions to process as comma-separated values (e.g., --repetitions 1,2,4,8). If not specified, process all available.'
    )
    
    return parser

def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 200_000
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def truncate_and_decode_swapped_data(tokenizer, args):
    input_dir = Path(args.input_swapped_dir)
    output_dir = Path(args.output_base_dir) / "swapped_part"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Architecture-specific cache
    arch = platform.machine()
    cache_dir = f"/iopsstor/scratch/cscs/xyixuan/cache/{arch}"
    
    rep_files = sorted(input_dir.glob("rep_*_swaps_token.jsonl"))
    
    # Filter by repetitions if specified
    if args.repetitions:
        repetitions_set = set(int(rep) for rep in args.repetitions.split(','))
        filtered_files = []
        for rep_file in rep_files:
            rep_num = int(rep_file.stem.split('_')[1])
            if rep_num in repetitions_set:
                filtered_files.append(rep_file)
        rep_files = filtered_files
    
    print(f"\n{'='*60}")
    print(f"Processing Swapped Data")
    print(f"{'='*60}")
    print(f"Found {len(rep_files)} swapped token files to process")
    if args.repetitions:
        print(f"Processing only repetitions: {args.repetitions}")
    print(f"Truncating to first {args.truncate_length} tokens")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {args.num_proc or 'auto'}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    def process_batch(batch):
        """Process a batch of samples using HuggingFace's batching."""
        truncated_ids_list = []
        original_lengths = []
        
        for input_ids in batch['input_ids']:
            truncated_ids = input_ids[:args.truncate_length]
            truncated_ids_list.append(truncated_ids)
            original_lengths.append(len(input_ids))
        
        # Batch decode all at once
        decoded_texts = tokenizer.batch_decode(truncated_ids_list, skip_special_tokens=True)
        
        return {
            'text': decoded_texts,
            'token_length': [len(ids) for ids in truncated_ids_list],
            'original_length': original_lengths
        }
    
    for i, rep_file in enumerate(tqdm(rep_files, desc="Processing swapped files")):
        rep_num = rep_file.stem.split('_')[1]
        output_file = output_dir / f"rep_{rep_num}_text_{args.truncate_length}.jsonl"
        
        # Check if already exists
        if output_file.exists() and not args.overwrite:
            print(f"\n[{i+1}/{len(rep_files)}] Skipping rep_{rep_num} - already exists")
            continue
        
        print(f"\n[{i+1}/{len(rep_files)}] Loading rep_{rep_num} from {rep_file.name}")
        
        dataset = load_dataset(
            'json',
            data_files=str(rep_file),
            split='train',
            cache_dir=cache_dir
        )
        
        print(f"  - Loaded {len(dataset)} samples")
        print(f"  - Starting batch decoding...")
        
        # Use HuggingFace's map with batching and parallel processing
        processed_dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            desc=f"  Decoding rep_{rep_num}"
        )
        
        # Save using HuggingFace's to_json
        print(f"  - Saving to {output_file.name}")
        processed_dataset.to_json(str(output_file))
        
        print(f"  ✓ Completed rep_{rep_num}")

def process_inference_suffixes(tokenizer, args):
    # Input should be a specific offset directory
    offset_dir = Path(args.input_inference_dir)
    
    if not offset_dir.name.startswith("offset_"):
        raise ValueError(f"Expected a specific offset directory (e.g., offset_0_prefix_500_suffix_500), got: {offset_dir.name}")
    
    # Extract base directory (parent of offset directory)
    base_dir = offset_dir.parent
    
    output_base = Path(args.output_base_dir) / "suffix" / args.model_size
    
    offset_info = offset_dir.name  
    
    # Parse offset, prefix, suffix values
    offset_parts = offset_info.split('_')
    offset_val = offset_parts[1]
    prefix_val = offset_parts[3]
    suffix_val = offset_parts[5]
    
    output_dir = output_base / f"offset_{offset_val}_prefix_{prefix_val}_suffix_{suffix_val}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing Generated Suffixes")
    print(f"{'='*60}")
    print(f"Offset directory: {offset_info}")
    print(f"Model size: {args.model_size}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {args.num_proc or 'auto'}")
    print(f"Batch size: {args.batch_size}")
    
    # Process all greedy generation results
    rep_dirs = sorted(offset_dir.glob("rep_*_greedy"))
    
    # Filter by repetitions if specified
    if args.repetitions:
        repetitions_set = set(int(rep) for rep in args.repetitions.split(','))
        filtered_dirs = []
        for rep_dir in rep_dirs:
            rep_num = int(rep_dir.name.split('_')[1])
            if rep_num in repetitions_set:
                filtered_dirs.append(rep_dir)
        rep_dirs = filtered_dirs
    
    print(f"Found {len(rep_dirs)} rep directories to process")
    if args.repetitions:
        print(f"Processing only repetitions: {sorted(repetitions_set)}")
    print(f"{'='*60}\n")
    
    for i, rep_dir in enumerate(tqdm(rep_dirs, desc=f"Processing {offset_info}")):
            rep_info = rep_dir.name  
            rep_parts = rep_info.split('_')
            rep_num = rep_parts[1]
            policy = rep_parts[2]
            
            try:
                print(f"\n[{i+1}/{len(rep_dirs)}] Processing rep_{rep_num}_{policy}")
                print(f"  - Loading suffix data...")
                
                # Load suffix data using existing utility
                dataset = load_inference_data(
                    base_dir=base_dir,
                    offset=int(offset_val),
                    len_prefix=int(prefix_val),
                    len_suffix=int(suffix_val),
                    rep=int(rep_num),
                    policy=policy
                )
                
                print(f"  - Loaded {len(dataset)} samples")
                
                output_file = output_dir / f"rep_{rep_num}_{policy}_text.jsonl"
                
                # Skip if already exists
                if output_file.exists() and not args.overwrite:
                    print(f"  - Skipping, already exists: {output_file.name}")
                    continue
                
                def decode_batch(batch):
                    """Batch decode generated suffixes to text only."""
                    # Batch decode all suffixes at once
                    decoded_texts = tokenizer.batch_decode(batch['generated_suffix'], skip_special_tokens=True)
                    
                    return {
                        'text': decoded_texts  # Only return text for decontamination
                    }
                
                print(f"  - Starting batch decoding...")
                
                # Process with batching and parallel processing
                processed_dataset = dataset.map(
                    decode_batch,
                    batched=True,
                    batch_size=args.batch_size,
                    num_proc=args.num_proc,
                    desc=f"  Decoding rep_{rep_num}",
                    remove_columns=dataset.column_names  # Remove all original columns
                )
                
                # Save processed dataset
                print(f"  - Saving to {output_file.name}")
                processed_dataset.to_json(str(output_file))
                
                print(f"  ✓ Completed rep_{rep_num}_{policy}")
                
            except Exception as e:
                print(f"  ✗ Error processing {rep_dir}: {e}")
                continue


def build_indices_for_swapped_parts(args):
    """Build indices for all swapped part files in parallel."""
    swapped_dir = Path(args.output_base_dir) / "swapped_part"
    output_dir = Path(args.indices_output_dir) / "indices" / "swapped_part"
    
    if not swapped_dir.exists():
        print(f"Swapped part directory not found: {swapped_dir}")
        return
    
    jsonl_files = sorted(swapped_dir.glob(f"rep_*_text_{args.truncate_length}.jsonl"))
    
    # Filter by repetitions if specified
    if args.repetitions:
        repetitions_set = set(int(rep) for rep in args.repetitions.split(','))
        filtered_files = []
        for file_path in jsonl_files:
            rep_num = int(file_path.stem.split('_')[1])
            if rep_num in repetitions_set:
                filtered_files.append(file_path)
        jsonl_files = filtered_files
    
    print(f"\nFound {len(jsonl_files)} swapped part files to index")
    if args.repetitions:
        print(f"Processing only repetitions: {sorted(repetitions_set)}")
    
    # Determine number of parallel workers for index building
    # Limit to 8 workers max to avoid memory issues with large files
    max_index_workers = 8
    index_workers = min(args.num_proc or cpu_count(), len(jsonl_files), max_index_workers)
    print(f"Using {index_workers} parallel workers for index building (max: {max_index_workers})")
    
    # Prepare arguments for parallel processing
    build_args = [{
        'file_path': file_path,
        'output_dir': output_dir,
        'n_gram_size': args.n_gram_size,
        'text_key': 'text',
        'overwrite': args.overwrite
        # limit defaults to -1 in build_index_wrapper
    } for file_path in jsonl_files]
    
    # Process files in parallel
    with Pool(index_workers) as pool:
        results = list(tqdm(
            pool.imap(build_index_wrapper, build_args),
            total=len(jsonl_files),
            desc="Building indices for swapped parts"
        ))
    
    # Report results
    errors = [r for r in results if r.startswith("Error:")]
    skipped = [r for r in results if r.startswith("Skipped:")]
    success = [r for r in results if r.startswith("Success:")]
    
    print(f"\nIndex building complete: {len(success)} created, {len(skipped)} skipped, {len(errors)} errors")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  {error}")

def build_indices_for_inference_suffixes(args):
    """Build indices for generated suffix files in parallel."""
    # Parse the offset info from the input dir to know which output to process
    offset_dir_input = Path(args.input_inference_dir)
    if offset_dir_input.name.startswith("offset_"):
        offset_info = offset_dir_input.name
    else:
        print("Warning: --input-inference-dir should be a specific offset directory for index building")
        return
    
    suffix_dir = Path(args.output_base_dir) / "suffix" / args.model_size / offset_info
    output_dir = Path(args.indices_output_dir) / "indices" / "suffix" / args.model_size / offset_info
    
    if not suffix_dir.exists():
        print(f"Suffix directory not found: {suffix_dir}")
        return
    
    # Process all rep files in this specific offset directory
    jsonl_files = sorted(suffix_dir.glob("rep_*_*_text.jsonl"))
    
    # Filter by repetitions if specified
    if args.repetitions:
        repetitions_set = set(int(rep) for rep in args.repetitions.split(','))
        filtered_files = []
        for file_path in jsonl_files:
            rep_num = int(file_path.stem.split('_')[1])
            if rep_num in repetitions_set:
                filtered_files.append(file_path)
        jsonl_files = filtered_files
    
    print(f"Found {len(jsonl_files)} files to index in {suffix_dir}")
    if args.repetitions:
        print(f"Processing only repetitions: {sorted(repetitions_set)}")
    
    # Determine number of parallel workers
    # Limit to 8 workers max to avoid memory issues with large files
    max_index_workers = 8
    index_workers = min(args.num_proc or cpu_count(), len(jsonl_files), max_index_workers)
    print(f"Using {index_workers} parallel workers for index building (max: {max_index_workers})")
    
    # Prepare arguments for parallel processing
    build_args = [{
        'file_path': file_path,
        'output_dir': output_dir,
        'n_gram_size': args.n_gram_size,
        'text_key': 'text',
        'overwrite': args.overwrite
        # limit defaults to -1 in build_index_wrapper
    } for file_path in jsonl_files]
    
    # Process files in parallel
    with Pool(index_workers) as pool:
        results = list(tqdm(
            pool.imap(build_index_wrapper, build_args),
            total=len(jsonl_files),
            desc=f"Building indices for {offset_info}"
        ))
    
    # Report results
    errors  = [r for r in results if r.startswith("Error:")]
    skipped = [r for r in results if r.startswith("Skipped:")]
    success = [r for r in results if r.startswith("Success:")]
    
    print(f"\nIndex building complete: {len(success)} created, {len(skipped)} skipped, {len(errors)} errors")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  {error}")

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Only process what is explicitly requested
    # If no processing flags are set, only build indices will run (if --build-indices is set)
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(args.tokenizer_model)
    
    if args.process_swapped:
        print("=" * 50)
        print("Processing swapped Gutenberg data...")
        print("=" * 50)
        truncate_and_decode_swapped_data(tokenizer, args)
    
    if args.process_suffix:
        print("\n" + "=" * 50)
        print("Processing inference suffixes...")
        print("=" * 50)
        process_inference_suffixes(tokenizer, args)
    
    # Build indices if requested
    if args.build_indices:
        print("\n" + "=" * 50)
        print("Building n-gram indices...")
        print("=" * 50)
        
        if args.process_swapped:
            build_indices_for_swapped_parts(args)
        
        if args.process_suffix:
            build_indices_for_inference_suffixes(args)
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main()