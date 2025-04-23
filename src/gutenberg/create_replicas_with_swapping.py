from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import gc
import logging
import os
import numpy as np
import random
from typing import List
import argparse

# Import the config
from config import DataConfig, FILE_NAMES
from create_replicas import load_and_validate_data
from create_excerpt import create_tokenize_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('replicate_with_swapping.log'),
        logging.StreamHandler()
    ]
)

_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
_tokenizer.model_max_length = 200_000
_tokenizer.pad_token_id = _tokenizer.eos_token_id
tokenize_fn = create_tokenize_fn(_tokenizer)

def setup_arg_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='Create replicated datasets with token swapping')
    parser.add_argument(
        '--attempts-per-sequence', 
        type=int, 
        default=5,
        help='Maximum number of attempts per sequence (default: 5)'
    )
    parser.add_argument(
        '--split-point',
        type=int,
        default=4000,
        help='Split point for token swapping (default: 4000)'
    )
    parser.add_argument(
        '--early-termination',
        type=int,
        default=50,
        help='Number of consecutive failed attempts before skipping a sequence (default: 50)'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=42,
        help='Base random seed (default: 42)'
    )

    return parser

def detokenize(tokens):
    """Convert tokens back to text"""
    return _tokenizer.decode(tokens)

def tokenize(text):
    """Convert text to tokens"""
    return tokenize_fn(text)["input_ids"]

def verify_token_consistency(original_tokens: List[int], new_tokens: List[int], split_point: int) -> bool:
    """
    Verify that the first split_point tokens and the remaining tokens match their respective sources
    """
    # Check if the lengths are consistent first
    if len(original_tokens) != len(new_tokens):
        return False
    
    # Check if the last (len-split_point) tokens are identical
    return original_tokens[split_point:] == new_tokens[split_point:]


def process_sequence_swap(source_tokens, swap_tokens, split_point, sample_idx):
    """Process a single token swap and validate it."""
    # Create new swapped tokens
    new_tokens = swap_tokens[:split_point] + source_tokens[split_point:]
    
    # Verify consistency (make sure the last 4K tokens match the source)
    if not verify_token_consistency(source_tokens, new_tokens, split_point):
        logging.warning(f"Token inconsistency detected for sample {sample_idx}. Skipping...")
        return None
    
    # Decode and re-tokenize to verify consistency
    try:
        new_text = detokenize(new_tokens)
        re_tokenized = tokenize(new_text)
        
        # Check if the re-tokenization preserves the tokens
        if not verify_token_consistency(new_tokens, re_tokenized, split_point):
            logging.warning(f"Re-tokenization inconsistency for sample {sample_idx}. Skipping...")
            return None
        
        # Create the swapped sequence record
        return {
            "input_ids": new_tokens,
            "text": new_text,
        }
    except Exception as e:
        logging.error(f"Error during detokenize/tokenize for sample {sample_idx}: {str(e)}")
        return None

def save_replicated_data_with_swapping(token, config: DataConfig, output_path: Path, attempts_per_sequence: int, split_point: int, early_termination: int, base_seed: int):
    """
    Save replicated datasets with swapping of first 5K tokens:
    1. Keep the last tokens intact (after split_point)
    2. Swap the first tokens with other sequences according to repetition config
    3. Validate tokenization consistency
    4. Generate on the fly and ensure exactly 500 sequences per bucket
    """

    # Find already processed repetitions and their sample indices
    completed_reps = {}
    sample_index_file = output_path / "sample_indices.txt"
    
    # Load previously processed indices if they exist
    if sample_index_file.exists():
        with open(sample_index_file, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    rep, index = parts
                    completed_reps[int(rep)] = int(index)
    
    # Check for completed repetitions without indices
    for path in output_path.glob("rep_*_token.jsonl"):
        rep = int(path.stem.split('_')[1])
        if rep not in completed_reps:
            # We found a repetition without a logged index, so we'll need to reprocess it
            logging.warning(f"Found repetition {rep} without sample index information. It may be reprocessed.")

    # Total number of samples in the token dataset
    total_samples = len(token)
    logging.info(f"Total samples available for swapping: {total_samples}")
    
    # Start from 0 or the last recorded sample offset
    sample_offset = 0
    last_recorded_rep = max(completed_reps.keys()) if completed_reps else 0
    if last_recorded_rep > 0:
        sample_offset = completed_reps[last_recorded_rep]
        logging.info(f"Resuming from sample offset {sample_offset} (after repetition {last_recorded_rep})")
    
    # Process each repetition
    for rep in tqdm(config.repetitions(), desc="Processing repetitions"):
        if rep in completed_reps:
            logging.info(f"Skipping repetition {rep} - already processed")
            continue

        # Set repetition-specific seed
        rep_seed = base_seed + rep
        logging.info(f"Using seed {rep_seed} for repetition {rep}")
        random.seed(rep_seed)
        np.random.seed(rep_seed)
        
        # Prepare containers for results
        tokens_to_save = []  # For storing last tokens
        swapped_sequences_tokens = []  # For storing swapped token sequences
        swapped_sequences_texts = []   # For storing swapped text sequences
        
        # Process until we have exactly 500 valid sequences
        valid_count = 0
        current_sample_idx = 0
        
        while valid_count < config.bucket_size and sample_offset + current_sample_idx < total_samples:
            # Get the source tokens
            source_tokens = token[sample_offset + current_sample_idx]["input_ids"]
            
            # For rep==1, no actual swapping needed
            if rep == 1:
                # Store the source tokens (last part) and the complete sequence
                tokens_to_save.append({"input_ids": source_tokens[split_point:]})
                swapped_sequences_tokens.append({"input_ids": source_tokens})
                swapped_sequences_texts.append({"text": detokenize(source_tokens)})
                valid_count += 1
                current_sample_idx += 1
                continue
                
            # For rep > 1, try to find valid swaps
            # Flag to track if this sample is valid (has all required swaps)
            sample_valid = True
            current_swapped_tokens = []
            current_swapped_texts = []
            
            # Create a list of all possible indices except the current one
            all_indices = list(range(total_samples))
            current_absolute_idx = sample_offset + current_sample_idx
            if current_absolute_idx in all_indices:
                all_indices.remove(current_absolute_idx)
            
            # Determine how many swaps we need for this repetition
            num_swaps = rep 

            # Prepare for collecting swaps
            collected_swaps = 0
            attempted_indices = set()

            # Try to find valid swaps
            while collected_swaps < num_swaps:
                # Get indices we haven't tried yet
                available_indices = [i for i in all_indices if i not in attempted_indices]

                # Check for early termination - if we've tried a certain number with zero success, skip
                if len(attempted_indices) >= early_termination and collected_swaps == 0:
                    logging.warning(f"Early termination: {early_termination} attempts with ZERO successful swaps. Skipping sequence {current_sample_idx}.")
                    sample_valid = False
                    break

                # Break if we've tried too many times (rep*1000 attempts)
                max_attempt = min(len(all_indices), num_swaps*attempts_per_sequence)
                if len(attempted_indices) >= max_attempt:
                    logging.warning(f"Exceeded maximum attempts ({max_attempt}) for sample {current_sample_idx}. Only collected {collected_swaps}/{num_swaps} swaps. Skip this sequence")
                    sample_valid = False
                    break

                # Sample a new index
                swap_idx = random.choice(available_indices)
                attempted_indices.add(swap_idx)
                
                # Get swap tokens directly from the token dataset
                swap_tokens = token[swap_idx]["input_ids"]
                
                # Process the swap
                result = process_sequence_swap(
                    source_tokens, swap_tokens, split_point, current_sample_idx
                )
                
                # If successful, add to our collection
                if result:
                    current_swapped_tokens.append({"input_ids": result["input_ids"]})
                    current_swapped_texts.append({"text": result["text"]})
                    collected_swaps += 1
            
            # Only include this sample if all swaps were successful
            if sample_valid:
                tokens_to_save.append({"input_ids": source_tokens[split_point:]})
                swapped_sequences_tokens.extend(current_swapped_tokens)
                swapped_sequences_texts.extend(current_swapped_texts)
                valid_count += 1
            
            # Move to next sample
            current_sample_idx += 1
            
            # Check if we're approaching the end of the dataset
            if sample_offset + current_sample_idx >= total_samples and valid_count < config.bucket_size:
                logging.error(f"Reached the end of the dataset with only {valid_count}/500 valid samples for rep_{rep}.")
                break
        
        # Save datasets if we have enough samples
        if valid_count == config.bucket_size:
            # Save token dataset
            token_dataset = Dataset.from_list(tokens_to_save)
            token_path = output_path / f"rep_{rep}_{FILE_NAMES['TOKEN']}"
            token_dataset.to_json(str(token_path))
            logging.info(f"Saved {len(token_dataset)} token sequences to {token_path}")
            
            # Save swapped tokens
            swaps_token_dataset = Dataset.from_list(swapped_sequences_tokens)
            swaps_token_path = output_path / f"rep_{rep}_swaps_token.jsonl"
            swaps_token_dataset.to_json(str(swaps_token_path))
            logging.info(f"Saved {len(swaps_token_dataset)} swapped token sequences to {swaps_token_path}")
            
            # Save swapped texts
            swaps_text_dataset = Dataset.from_list(swapped_sequences_texts)
            swaps_text_path = output_path / f"rep_{rep}_swaps_text.jsonl"
            swaps_text_dataset.to_json(str(swaps_text_path))
            logging.info(f"Saved {len(swaps_text_dataset)} swapped text sequences to {swaps_text_path}")
            
            # Update and log the new sample offset
            new_offset = sample_offset + current_sample_idx
            completed_reps[rep] = new_offset
            
            # Log to file so we can resume if interrupted
            with open(sample_index_file, 'w') as f:
                for r, idx in completed_reps.items():
                    f.write(f"{r}:{idx}\n")
            
            logging.info(f"Completed repetition {rep}. Next sample offset: {new_offset}")
            
            # Update sample offset for next repetition
            sample_offset = new_offset
        else:
            logging.error(f"Failed to find 500 valid samples for rep_{rep}. Only found {valid_count}. Skipping.")
            
            # Even for failures, we should record where we stopped
            new_offset = sample_offset + current_sample_idx
            logging.info(f"Failed repetition {rep} stopped at sample offset: {new_offset}")
            sample_offset = new_offset
        
        # Check if we've processed the entire dataset
        if sample_offset >= total_samples:
            logging.warning(f"Reached the end of the dataset after processing rep_{rep}.")
            break
            
        # Clean up
        gc.collect()


def main():
    # Set up argument parser
    parser = setup_arg_parser()
    args = parser.parse_args()

    config = DataConfig()
    input_path = Path('/capstor/users/cscs/xyixuan/data/raw/gutenberg_en_8k') 
    output_path = Path("/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_swapped")
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading dataset...")
    token_seq = load_and_validate_data(config, input_path / FILE_NAMES['TOKEN'])
    
    logging.info("Saving replicated datasets with token swapping...")
    save_replicated_data_with_swapping(
        token=token_seq,
        config=config, 
        output_path=output_path,
        attempts_per_sequence=args.attempts_per_sequence,
        split_point=args.split_point,
        early_termination=args.early_termination,
        base_seed=args.base_seed
    )
    
    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()