#!/usr/bin/env python3
"""
Retokenize Gutenberg dataset from Llama to Mixtral tokenization using HuggingFace datasets.

This script:
1. Reads text sequences from llama_reps (already 8191 Llama tokens)
2. Re-tokenizes the text using Mixtral tokenizer
3. Saves both text and tokens to mixtral_reps
"""

import json
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from config import DataConfig, FILE_NAMES

class TokenizationStats:
    """Track tokenization statistics efficiently."""
    
    def __init__(self):
        self.mixtral_stats = defaultdict(int)
        self.sequence_records = []  # Store (rep, seq_id, token_count) tuples
    
    def update(self, mixtral_count):
        """Update statistics for Mixtral tokenizer."""
        self.mixtral_stats[mixtral_count] += 1
    
    def add_sequence(self, rep, seq_id, token_count):
        """Record token count for a specific sequence."""
        self.sequence_records.append((rep, seq_id, token_count))
    
    def print_summary(self):
        """Print comprehensive statistics summary."""
        if not self.mixtral_stats:
            return
        
        print("\n" + "="*60)
        print("MIXTRAL TOKENIZATION STATISTICS")
        print("="*60)
        
        # Mixtral statistics
        self._print_tokenizer_stats("Mixtral", self.mixtral_stats)
    
    def _print_tokenizer_stats(self, name, stats):
        """Print statistics for a single tokenizer."""
        total = sum(stats.values())
        unique_counts = len(stats)
        
        print(f"\n{name} Tokenizer ({total:,} sequences):")
        print("-" * 40)
        
        # If all sequences have the same token count
        if unique_counts == 1:
            token_count = list(stats.keys())[0]
            print(f"✓ All sequences: {token_count} tokens")
        else:
            # Show distribution
            print(f"Token count distribution ({unique_counts} unique counts):")
            for token_count in sorted(stats.keys()):
                count = stats[token_count]
                percentage = 100 * count / total
                bar = "█" * int(percentage / 2)  # Simple bar chart
                print(f"  {token_count:5d} tokens: {count:5d} seqs ({percentage:5.1f}%) {bar}")
            
            # Statistics
            tokens_list = []
            for token_count, freq in stats.items():
                tokens_list.extend([token_count] * freq)
            
            avg_tokens = sum(tokens_list) / len(tokens_list)
            min_tokens = min(stats.keys())
            max_tokens = max(stats.keys())
            
            print(f"\nStatistics:")
            print(f"  Average: {avg_tokens:.1f} tokens")
            print(f"  Range: {min_tokens} - {max_tokens} tokens")
            print(f"  Spread: {max_tokens - min_tokens} tokens")
    
    def save_to_file(self, output_path):
        """Save token counts to a TSV file."""
        with open(output_path, 'w') as f:
            # Write header
            f.write("rep\tseq_id\ttoken_count\n")
            
            # Write records
            for rep, seq_id, token_count in self.sequence_records:
                f.write(f"rep_{rep}\t{seq_id}\t{token_count}\n")
        
        print(f"\nToken counts saved to: {output_path}")

def retokenize_with_datasets():
    """Retokenize all rep files from Llama to Mixtral using HuggingFace datasets."""
    
    config = DataConfig()
    
    # Define paths
    input_dir = Path("/capstor/store/cscs/swissai/infra01/users/xyixuan/dataset/memorization_studies/gutenberg_en_8190_llama_to_mixtral/llama_reps")
    output_dir = Path("/capstor/store/cscs/swissai/infra01/users/xyixuan/dataset/memorization_studies/gutenberg_en_8190_llama_to_mixtral/mixtral_reps")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n" + "="*60)
    print("INITIALIZING TOKENIZER")
    print("="*60)
    print("Loading Mixtral tokenizer...")
    mixtral_tokenizer = AutoTokenizer.from_pretrained("alehc/swissai-tokenizer")
    print("✓ Tokenizer loaded successfully")
    
    # Initialize statistics tracker
    stats = TokenizationStats()
    
    # Track processing
    processed_reps = []
    skipped_reps = []
    
    print("\n" + "="*60)
    print("PROCESSING REPETITIONS")
    print("="*60)
    
    for rep in config.repetitions():
        input_file = input_dir / f"rep_{rep}_{FILE_NAMES['TEXT']}"
        output_text_file = output_dir / f"rep_{rep}_{FILE_NAMES['TEXT']}"
        output_token_file = output_dir / f"rep_{rep}_{FILE_NAMES['TOKEN']}"
        
        # Check if input exists
        if not input_file.exists():
            skipped_reps.append(f"rep_{rep} (input not found)")
            continue
        
        # Check if outputs already exist
        if output_text_file.exists() and output_token_file.exists():
            skipped_reps.append(f"rep_{rep} (output exists)")
            continue
        
        print(f"\nProcessing rep_{rep}:")
        print("-" * 30)
        
        # Load dataset
        dataset = load_dataset('json', data_files=str(input_file), split='train')
        print(f"  Loaded: {len(dataset)} sequences")
        
        # Track sequence counter for this rep
        seq_counter = 0
        
        # Define tokenization function
        def tokenize_batch(examples):
            """Tokenize with Mixtral tokenizer."""
            nonlocal seq_counter
            results = {'input_ids': []}
            
            for text in examples['text']:
                # Retokenize with Mixtral
                mixtral_tokens = mixtral_tokenizer.encode(text, add_special_tokens=False)
                results['input_ids'].append(mixtral_tokens)
                
                # Update statistics
                token_count = len(mixtral_tokens)
                stats.update(token_count)
                stats.add_sequence(rep, seq_counter, token_count)
                
                seq_counter += 1
            
            return results
        
        # Apply tokenization
        print(f"  Retokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=100,  # Process in batches of 100
            desc=f"    Progress",
            remove_columns=[]  # Keep all columns
        )
        
        # Save files
        print(f"  Saving files...")
        
        # Save text (original dataset)
        dataset.to_json(str(output_text_file))
        
        # Save tokens (only the input_ids column)
        tokenized_dataset.select_columns(['input_ids']).to_json(str(output_token_file))
        
        print(f"  ✓ Completed")
        processed_reps.append(f"rep_{rep}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    if processed_reps:
        print(f"\nProcessed ({len(processed_reps)} files):")
        for rep in processed_reps:
            print(f"  ✓ {rep}_text.jsonl & {rep}_token.jsonl")
    
    if skipped_reps:
        print(f"\nSkipped ({len(skipped_reps)} files):")
        for rep in skipped_reps:
            print(f"  - {rep}")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Print tokenization statistics
    stats.print_summary()
    
    # Save token counts to file
    token_counts_file = output_dir / "mixtral_token_counts.tsv"
    stats.save_to_file(token_counts_file)
    
    print("\n" + "="*60)
    print("✓ RETOKENIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retokenize all rep files from Llama to Mixtral tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s                     # Use default paths
        %(prog)s --batch-size 200    # Larger batch size for faster processing
        """
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Run the retokenization
    retokenize_with_datasets()