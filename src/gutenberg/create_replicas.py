import numpy as np
import json
from datasets import load_dataset
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataConfig:
    fineweb_edu_size: int = 81_816_372_499
    bucket_size: int = 500
    seq_length: int = 8192
    repetitions: np.ndarray = np.array([1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128])

    def get_total_tokens(self) -> str:
        """Calculate and return total number of tokens in billions."""
        total_tokens = np.sum(self.repetitions * self.bucket_size * self.seq_length)
        return f"{(total_tokens / 1e9):.2f}B"
    
    def __post_init__(self):
        """Print total tokens after initialization."""
        print(f"Total tokens in dataset: {self.get_total_tokens()}")


class GutenbergSampler:
    def __init__(self, dataset, bucket_size: int):
        self.dataset = dataset
        self.bucket_size = bucket_size
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
    def sample_bucket(self, start_idx: int):
        """Sample a bucket of data without replacement using select."""
        return self.dataset.select(range(start_idx, start_idx + self.bucket_size))

def load_and_validate_data(config: DataConfig, data_path: Path):
    dataset = load_dataset('json', data_files=str(data_path), split='train')
    assert len(config.repetitions) * config.bucket_size <= len(dataset), (
        f"Required {len(config.repetitions) * config.bucket_size} samples, "
        f"but dataset only contains {len(dataset)}"
    )
    return dataset

def save_replicated_data(sampler: GutenbergSampler, config: DataConfig, base_path: Path):
    """Save replicated datasets using sampling without replacement."""
    for idx, rep in enumerate(config.repetitions):
        # Sample current bucket
        start_idx = idx * config.bucket_size
        current_bucket = sampler.sample_bucket(start_idx)
        
        # Save replicated data
        output_file = base_path / f"rep_{rep}_tokens.jsonl"
        with open(output_file, 'w') as f:
            for _ in range(rep):
                for item in current_bucket:
                    f.write(json.dumps(item) + '\n')
        
        print(f"Saved {rep} repetitions to {output_file}")

def main():
    config = DataConfig()
    base_path = Path('/capstor/users/cscs/xyixuan/data/raw/gutenberg_en_8k')
    data_path = base_path / 'tokenized.jsonl'
    
    # Load dataset
    dataset = load_and_validate_data(config, data_path)
    
    # Initialize sampler
    sampler = GutenbergSampler(dataset, config.bucket_size)
    
    # Save replicated datasets
    save_replicated_data(sampler, config, base_path)

if __name__ == "__main__":
    main()