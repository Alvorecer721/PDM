from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import gc
import logging
import os
import numpy as np

# Import the config
from config import DataConfig, FILE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('replicate_gutenberg_excerpt.log'),
        logging.StreamHandler()
    ]
)

def load_and_validate_data(config: DataConfig, data_path: Path):
    dataset = load_dataset('json', data_files=str(data_path), split='train')
    assert len(config.repetitions()) * config.bucket_size <= len(dataset), (
        f"Required {len(config.repetitions()) * config.bucket_size} samples, "
        f"but dataset only contains {len(dataset)}"
    )
    return dataset

def save_replicated_data(text, token, config: DataConfig, output_path: Path):
    """Save replicated datasets using HuggingFace's efficient methods."""

    # Check for existing files
    completed_reps = set()
    for path in output_path.glob("rep_*_text.jsonl"):
        rep = int(path.stem.split('_')[1])
        completed_reps.add(rep)

    for idx, rep in enumerate(tqdm(config.repetitions(), desc="Processing buckets")):
        if rep in completed_reps:
            logging.info(f"Skipping repetition {rep} - already processed")
            continue

        # Get current slice
        start_idx = idx * config.bucket_size
        current_slice_text = text.select(range(start_idx, start_idx + config.bucket_size))
        current_slide_token = token.select(range(start_idx, start_idx + config.bucket_size)) # token does not require replication
        
        # Create replicated version efficiently
        if rep > 1:
            replicated_slices = [current_slice_text] * rep
            current_slice_text = concatenate_datasets(replicated_slices)
        
        # Save using dataset's built-in method
        output_text_file = str(output_path / f"rep_{rep}_{FILE_NAMES['TEXT']}")
        current_slice_text.to_json(output_text_file)
        current_slide_token.to_json(str(output_path / f"rep_{rep}_{FILE_NAMES['TOKEN']}"))
        
        # logging.info(f"Saved {rep} repetitions ({len(current_slice_text)} samples) to {output_text_file}")

        # Cleanup
        current_slice_text = None
        current_slide_token = None
        gc.collect()

def save_replicated_text_in_one(text, config: DataConfig, output_path: Path):
    """Save all replicated text data in a single JSON file with each sequence trimmed to 8190 tokens."""
    
    logging.info("Creating combined replicated dataset...")
    
    all_replicated_text = []
    
    for idx, rep in enumerate(tqdm(config.repetitions(), desc="Combining text data")):
        # Get current slice
        start_idx = idx * config.bucket_size
        current_slice_text = text.select(range(start_idx, start_idx + config.bucket_size))
        
        # Create replicated version efficiently
        if rep > 1:
            replicated_slices = [current_slice_text] * rep
            current_slice_text = concatenate_datasets(replicated_slices)
        
        # Append to combined dataset
        all_replicated_text.append(current_slice_text)
        
        # Cleanup
        current_slice_text = None
        gc.collect()
    
    # Combine all slices into one dataset
    combined_dataset = concatenate_datasets(all_replicated_text)
    
    # Save combined dataset
    output_file = str(output_path / f"combined_gutenberg_text_{config.seq_length * config.bucket_size * np.sum(config.repetitions())}.jsonl")
    combined_dataset.to_json(output_file)
    
    logging.info(f"Saved combined replicated text dataset ({len(combined_dataset)} samples) to {output_file}")
    
    # Final cleanup
    all_replicated_text = None
    combined_dataset = None
    gc.collect()


# def main():
#     config = DataConfig()
#     input_path = Path('/capstor/users/cscs/xyixuan/data/raw/gutenberg_en_8k') 
#     output_path = Path("/iopsstor/scratch/cscs/xyixuan/dataset/gunteberg")
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     logging.info("Loading dataset...")
#     token_seq = load_and_validate_data(config, input_path / FILE_NAMES['TOKEN'])
#     text_seq  = load_and_validate_data(config, input_path / FILE_NAMES['TEXT'])
    
#     logging.info("Saving replicated datasets...")
#     save_replicated_data(
#         text=text_seq, 
#         token=token_seq, 
#         config=config, 
#         output_path=output_path
#     )
 
def main():
    config = DataConfig()
    input_path = Path('/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_en_8k_mixtral') 
    output_path = Path("/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_apertus_buk167")
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("Loading dataset...")
    token_seq = load_and_validate_data(config, input_path / FILE_NAMES['TOKEN'])
    text_seq = load_and_validate_data(config, input_path / FILE_NAMES['TEXT'])
    
    logging.info("Saving replicated datasets...")
    save_replicated_data(
        text=text_seq, 
        token=token_seq, 
        config=config, 
        output_path=output_path
    )
    
    # logging.info("Saving all replicated text in one file...")
    # save_replicated_text_in_one(
    #     text=text_seq,
    #     config=config,
    #     output_path=output_path
    # )
    
    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()