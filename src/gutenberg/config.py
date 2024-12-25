from dataclasses import dataclass
import numpy as np
import logging

# Constants
COLUMN_NAMES = {
    "TEXT": "text",
    "ID": "id",
    "INPUT_IDS": "input_ids",
    "SELECTED_TOKENS": "selected_tokens",
    "DETOKENIZED_TEXTS": "detokenized_texts",
    "SEQ_LENGTH": "seq_length",
}

FILE_NAMES = {"TOKEN": "token.jsonl", "TEXT": "text.jsonl"}

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
        logging.info(f"Total tokens in dataset: {self.get_total_tokens()}")