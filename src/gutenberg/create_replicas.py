from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import gc
import logging
import os

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
    assert len(config.repetitions) * config.bucket_size <= len(dataset), (
        f"Required {len(config.repetitions) * config.bucket_size} samples, "
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

    for idx, rep in enumerate(tqdm(config.repetitions, desc="Processing buckets")):
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


def main():
    config = DataConfig()
    input_path = Path('/capstor/users/cscs/xyixuan/data/raw/gutenberg_en_8k') 
    output_path = Path(os.getenv("SCRATCH")) / "dataset/gutenberg"
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Loading dataset...")
    token_seq = load_and_validate_data(config, input_path / FILE_NAMES['TOKEN'])
    text_seq  = load_and_validate_data(config, input_path / FILE_NAMES['TEXT'])
    
    logging.info("Saving replicated datasets...")
    save_replicated_data(
        text=text_seq, 
        token=token_seq, 
        config=config, 
        output_path=output_path
    )

if __name__ == "__main__":
    main()