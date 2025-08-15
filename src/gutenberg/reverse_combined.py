#!/usr/bin/env python3
"""
Extract unique sequences from the combined replicated file and save them by repetition count.
Verifies that each sequence is exactly 8190 Llama tokens.
"""

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from config import DataConfig, FILE_NAMES

def reverse_combined_file(output_dir, verify_tokens=True):
    """
    Extract unique sequences from the combined file and save as rep_*_text.jsonl files.
    
    The combined file has sequences replicated in ABCABC pattern:
    - Bucket 0: 500 sequences repeated 1 time = 500 total -> save as rep_1_text.jsonl
    - Bucket 1: 500 sequences repeated 2 times = 1000 total -> save as rep_2_text.jsonl
    - Bucket 2: 500 sequences repeated 3 times = 1500 total -> save as rep_3_text.jsonl
    - etc.
    """
    
    config = DataConfig()
    combined_file = Path("/capstor/store/cscs/swissai/a06/.NeMo/combined_gutenberg_text_1744683000.jsonl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Llama tokenizer for verification
    if verify_tokens:
        print("Loading Llama tokenizer for verification...")
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    print("Loading combined dataset...")
    combined_dataset = load_dataset('json', data_files=str(combined_file), split='train')
    print(f"Total sequences in combined file: {len(combined_dataset)}")
    
    current_position = 0
    total_extracted = 0
    token_count_stats = {}
    
    # Process each bucket
    for idx, rep in enumerate(tqdm(config.repetitions(), desc="Extracting buckets")):
        bucket_size = config.bucket_size
        
        # Check if this rep file already exists
        output_file = output_dir / f"rep_{rep}_{FILE_NAMES['TEXT']}"
        file_exists = output_file.exists()
        
        if file_exists:
            print(f"\nBucket {idx}: File exists for rep_{rep}")
            
            # Still verify token counts if requested and file exists
            if verify_tokens and idx < 3:  # Verify first 3 existing buckets
                print(f"  Verifying existing file token counts...")
                existing_dataset = load_dataset('json', data_files=str(output_file), split='train')
                
                sample_size = min(10, len(existing_dataset))
                for i in range(sample_size):
                    text = existing_dataset[i]['text']
                    tokens = llama_tokenizer.encode(text, add_special_tokens=False)
                    token_count = len(tokens)
                    
                    if token_count not in token_count_stats:
                        token_count_stats[token_count] = 0
                    token_count_stats[token_count] += 1
                    
                    if token_count != 8191 and token_count != 8190:
                        print(f"    Warning: Sequence {i} has {token_count} tokens")
            
            # Still need to advance position correctly
            replicated_block_size = bucket_size * rep
            current_position += replicated_block_size
            continue
        
        # The replicated block contains bucket_size sequences repeated rep times
        replicated_block_size = bucket_size * rep
        
        # Extract just the first bucket_size sequences from this replicated block
        # These are the unique sequences (before replication)
        end_position = current_position + bucket_size
        
        print(f"\nBucket {idx}: Extracting sequences {current_position}-{end_position-1} (rep={rep})")
        
        # Use HuggingFace's select to get the exact range we need
        unique_bucket = combined_dataset.select(range(current_position, end_position))
        
        # Verify token counts if requested
        if verify_tokens:
            print(f"  Verifying token counts for all {len(unique_bucket)} sequences...")
            
            for i in tqdm(range(len(unique_bucket)), desc=f"    Verifying", leave=False):
                text = unique_bucket[i]['text']
                tokens = llama_tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
                
                if token_count not in token_count_stats:
                    token_count_stats[token_count] = 0
                token_count_stats[token_count] += 1
                
                if token_count != 8191 and token_count != 8190:  # Show if not 8190 or 8191
                    print(f"    Warning: Sequence {i} has {token_count} tokens")
        
        # Save this bucket as rep_*_text.jsonl (output_file already defined above)
        unique_bucket.to_json(str(output_file))
        print(f"  Saved {len(unique_bucket)} sequences to {output_file.name}")
        
        total_extracted += len(unique_bucket)
        
        # Move to the start of the next replicated block
        current_position += replicated_block_size
    
    print(f"\nExtraction complete!")
    print(f"Total unique sequences extracted: {total_extracted}")
    print(f"Expected: {len(config.repetitions()) * config.bucket_size}")
    print(f"Output directory: {output_dir}")
    
    # Print token verification results if we did verification
    if verify_tokens and token_count_stats:
        print("\nLlama token verification (all sequences):")
        for token_count in sorted(token_count_stats.keys()):
            count = token_count_stats[token_count]
            print(f"  {token_count} tokens: {count} sequences")
        
        # Summary message based on results
        total_verified = sum(token_count_stats.values())
        if len(token_count_stats) == 1:
            token_count = list(token_count_stats.keys())[0]
            print(f"✓ All {total_verified} sequences have exactly {token_count} Llama tokens")
        else:
            print(f"⚠️ Llama token counts vary across {total_verified} sequences:")
            for token_count in sorted(token_count_stats.keys()):
                percentage = 100 * token_count_stats[token_count] / total_verified
                print(f"    {token_count} tokens: {percentage:.1f}%")
    
    # List all created files
    print("\nCreated files:")
    for rep in config.repetitions():
        text_file = output_dir / f"rep_{rep}_{FILE_NAMES['TEXT']}"
        if text_file.exists():
            print(f"  - {text_file.name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract unique sequences from combined replicated Gutenberg file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/users/xyixuan/dataset/memorization_studies/gutenberg_en_8190_llama_to_mixtral/llama_reps",
        nargs='?',
        help="Directory to save the extracted sequences as rep_*_text.jsonl files (default: %(default)s)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="/capstor/store/cscs/swissai/a06/.NeMo/combined_gutenberg_text_1744683000.jsonl",
        help="Path to the combined replicated file (default: %(default)s)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip token count verification with Llama tokenizer"
    )
    
    args = parser.parse_args()
    reverse_combined_file(args.output_dir, verify_tokens=not args.no_verify)