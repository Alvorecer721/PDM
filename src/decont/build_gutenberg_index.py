#!/usr/bin/env python3
"""
Build n-gram index for Gutenberg dataset for decontamination analysis.

This script builds n-gram indices for the Gutenberg dataset using DataTrove.
It uses the same commons functions as the FineWeb index building.

Usage:
    python build_gutenberg_index.py
    python build_gutenberg_index.py --n-gram-size 13
    python build_gutenberg_index.py --pilot-run  # Process only 1000 documents for testing
"""

import argparse
from pathlib import Path
import sys
import os

# Download NLTK data required by DataTrove
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab data...")
    nltk.download('punkt_tab', quiet=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.decont.commons import build_index_for_file


def main():
    parser = argparse.ArgumentParser(description='Build Gutenberg dataset index')
    
    parser.add_argument(
        '--input-file',
        default='/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_en_8k/text.jsonl',
        help='Path to the Gutenberg text.jsonl file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='/iopsstor/scratch/cscs/xyixuan/PDM/results/decont/indices/gutenberg',
        help='Output directory for indices'
    )
    
    parser.add_argument(
        '--n-gram-size',
        type=int,
        default=13,
        help='N-gram size for indexing (default: 13)'
    )
    
    parser.add_argument(
        '--pilot-run',
        action='store_true',
        help='Run in pilot mode - process only 1000 documents for testing'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=-1,
        help='Number of documents to process (-1 for all). Overridden by --pilot-run'
    )
    
    parser.add_argument(
        '--text-key',
        default='text',
        help='JSON key containing the text (default: "text")'
    )
    
    parser.add_argument(
        '--head',
        type=int,
        default=-1,
        help='Process only the first N samples from the dataset (-1 for all, e.g., 6000 for all repetition buckets)'
    )
    
    args = parser.parse_args()
    
    # If pilot run, set limit
    if args.pilot_run:
        args.limit = 1000
        print("\n" + "="*60)
        print("PILOT RUN MODE - Processing only 1000 documents")
        print("="*60 + "\n")
    elif args.head > 0:
        # Use head value directly as limit
        args.limit = args.head
        print(f"\nProcessing only first {args.head} samples from the dataset")
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    print(f"Building index for: {input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"N-gram size: {args.n_gram_size}")
    
    if args.limit > 0:
        print(f"Document limit: {args.limit}")
    
    # Check if index already exists
    output_folder = Path(args.output_dir)  # No subdirectory for Gutenberg
    index_file = output_folder / "input.index.hashes"
    stats_file = output_folder / "stats.json"
    
    if index_file.exists() and stats_file.exists():
        print(f"\nIndex already exists at: {output_folder}")
        print("To rebuild, remove the existing index or use a different output directory.")
        return
    
    try:
        # Build the index
        print("\nBuilding index...")
        result_path = build_index_for_file(
            file_path=input_path,
            output_dir=args.output_dir,
            n_gram_size=args.n_gram_size,
            text_key=args.text_key,
            limit=args.limit,
            use_subdirectory=False  # Save directly in gutenberg directory
        )
        
        print(f"\nIndex successfully built at: {result_path}")
        
        # Check output files
        if (Path(result_path) / "input.index.hashes").exists():
            print("✓ Hash index created")
        if (Path(result_path) / "stats.json").exists():
            print("✓ Statistics file created")
            
    except Exception as e:
        print(f"\nError building index: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()