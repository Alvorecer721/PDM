#!/usr/bin/env python3
"""
Optimized parallel index building for FineWeb chunks.

This script builds n-gram indices for FineWeb dataset chunks for decontamination analysis.
It processes chunks in parallel using multiprocessing.

Workflow:
1. Reads list of chunks from provided file
2. Checks which chunks already have indices (skips if exists)
3. Processes chunks in parallel using multiprocessing
4. Creates DataTrove n-gram indices with custom directory structure

Directory Structure:
- Input: /path/to/chunks/finewebedu_000001_chunk_000.jsonl
- Output: /path/to/indices/finewebedu_000001/chunk_000/input.index.hashes

Usage Examples:
--------------
# Basic usage:
python build_fineweb_indices_parallel.py --chunk-list chunks.txt

# Pilot run (process only 1000 documents per chunk):
python build_fineweb_indices_parallel.py --chunk-list chunks.txt --pilot-run

# Custom configuration:
python build_fineweb_indices_parallel.py \
    --chunk-list chunks.txt \
    --output-dir /path/to/indices \
    --workers 32 \
    --n-gram-size 13

# Process with specific document limit:
python build_fineweb_indices_parallel.py --chunk-list chunks.txt --limit 5000

Performance Notes:
- Adjust --workers based on available CPUs and memory
- Each worker processes one chunk at a time
- Pilot runs help test the pipeline before full processing
"""

import argparse
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import sys
import os
import tempfile
import shutil
from datatrove.executor import LocalPipelineExecutor

# Download NLTK data required by DataTrove
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab data...")
    nltk.download('punkt_tab', quiet=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.decont.commons import create_index_pipeline


def parse_filename(file_path):
    """Parse filename to extract base name and chunk name."""
    full_name = file_path.name.replace('.jsonl.gz', '').replace('.jsonl', '')
    
    # All files now follow format: finewebedu_XXXXXX_chunk_YYY
    parts = full_name.split('_')
    # Expected parts: ['finewebedu', 'XXXXXX', 'chunk', 'YYY']
    base_name = '_'.join(parts[:2])  # e.g., finewebedu_000003
    chunk_name = '_'.join(parts[2:])  # e.g., chunk_001
    
    return base_name, chunk_name


def build_fineweb_index(chunk_path, output_dir, n_gram_size=13, limit=-1):
    """Build n-gram index for a FineWeb chunk with custom directory structure."""
    try:
        chunk_path = Path(chunk_path)
        base_name, chunk_name = parse_filename(chunk_path)
        
        # FineWeb uses a nested directory structure: base_name/chunk_name/
        output_folder = Path(output_dir) / base_name / chunk_name
        
        # Check if already exists
        if (output_folder / "input.index.hashes").exists() and (output_folder / "stats.json").exists():
            return f"Skipped: {chunk_path.name} - index already exists"
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create unique temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"{chunk_name}_", dir=output_folder)
        
        try:
            # Create pipeline using the shared function
            pipeline = create_index_pipeline(
                chunk_path, 
                output_folder, 
                n_gram_size=n_gram_size, 
                text_key="text", 
                limit=limit
            )
            
            # Run pipeline
            executor = LocalPipelineExecutor(
                pipeline=pipeline,
                tasks=1,  # NGramsDecontIndexer only supports single worker
                logging_dir=temp_dir
            )
            executor.run()
            
            # Move output files from temp to final location
            stats_file = Path(temp_dir) / "stats.json"
            hash_file = Path(temp_dir) / "input.index.hashes"
            
            if stats_file.exists():
                shutil.move(str(stats_file), str(output_folder / "stats.json"))
            if hash_file.exists():
                shutil.move(str(hash_file), str(output_folder / "input.index.hashes"))
                
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return f"Success: {chunk_path.name}"
        
    except Exception as e:
        return f"Error: {chunk_path.name} - {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Build FineWeb indices in parallel')
    
    
    parser.add_argument(
        '--output-dir',
        default='/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/fineweb',
        help='Output directory for indices'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        required=True,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--n-gram-size',
        type=int,
        default=13,
        help='N-gram size for indexing'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing indices'
    )
    
    
    parser.add_argument(
        '--pilot-run',
        action='store_true',
        help='Run in pilot mode - process only 1000 documents per chunk for testing'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=-1,
        help='Number of documents to process per chunk (-1 for all). Overridden by --pilot-run'
    )
    
    parser.add_argument(
        '--chunk-list',
        type=str,
        required=True,
        help='File containing list of chunks to process (one per line)'
    )
    
    args = parser.parse_args()
    
    # If pilot run, set limit
    if args.pilot_run:
        args.limit = 1000
        print("\n" + "="*60)
        print("PILOT RUN MODE - Processing only 1000 documents per chunk")
        print("="*60 + "\n")
    
    # Read chunks from file
    if not args.chunk_list:
        print("Error: --chunk-list is required")
        sys.exit(1)
    
    with open(args.chunk_list, 'r') as f:
        chunk_paths = [Path(line.strip()) for line in f if line.strip()]
    
    # Filter out non-existent files
    chunks = [p for p in chunk_paths if p.exists()]
    
    # Optionally skip existing if not overwriting
    if not args.overwrite:
        chunks_to_process = []
        for chunk in chunks:
            base_name, chunk_name = parse_filename(chunk)
            index_path = Path(args.output_dir) / base_name / chunk_name / "input.index.hashes"
            if not index_path.exists():
                chunks_to_process.append(chunk)
        print(f"Found {len(chunks)} chunks, need to process {len(chunks_to_process)} chunks")
        chunks = chunks_to_process
    else:
        print(f"Processing {len(chunks)} chunks")
    
    if args.limit > 0 and not args.pilot_run:
        print(f"Limit: Processing only {args.limit} documents per chunk")
    
    if not chunks:
        print("All chunks already processed!")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process chunks in parallel
    # Adjust workers to not exceed the number of chunks to process
    actual_workers = min(args.workers, len(chunks))
    print(f"\nProcessing {len(chunks)} chunks with {actual_workers} workers...")
    
    # Prepare arguments for parallel processing
    process_args = [(chunk, args.output_dir, args.n_gram_size, args.limit) for chunk in chunks]
    
    with Pool(actual_workers) as pool:
        results = list(tqdm(
            pool.starmap(build_fineweb_index, process_args),
            total=len(chunks),
            desc="Building indices"
        ))
    
    # Summary
    successes = sum(1 for r in results if r.startswith("Success"))
    errors = [r for r in results if r.startswith("Error")]
    
    print(f"\nCompleted: {successes} successful, {len(errors)} errors")
    
    if errors:
        print("\nErrors:")
        for error in errors[:5]:
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()