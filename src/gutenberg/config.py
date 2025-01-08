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

# REPETITIONS = np.array([1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128])
REPETITIONS = np.array([2 ** i for i in range(7,12)])

@dataclass
class DataConfig:
    fineweb_edu_size: int = 81_816_372_499
    bucket_size: int = 500
    seq_length: int = 8192

    gutenberg_tokens = None
    total_tokens = None

    @classmethod
    def repetitions(cls) -> np.ndarray:
        return REPETITIONS
    
    def get_gutenberg_sequences(self) -> int:
        """Calculate and return total number of sequences in Gutenberg."""
        return np.sum(self.repetitions() * self.bucket_size)

    def get_gutenberg_tokens(self) -> str:
        """Calculate and return total number of tokens in billions."""
        self.gutenberg_tokens = np.sum( self.get_gutenberg_sequences() * self.seq_length )
        return f"{(self.gutenberg_tokens / 1e9):.2f}B"
    
    def get_total_tokens(self) -> str:
        """Calculate and return total number of tokens in billions."""
        self.total_tokens = self.gutenberg_tokens + self.fineweb_edu_size
        return f"{self.total_tokens / 1e9:.2f}B"
    
    def __post_init__(self):
        logging.info(f"Number of sequences in Gutenberg: {self.get_gutenberg_sequences()}")
        logging.info(f"Total tokens in Gutenberg: {self.get_gutenberg_tokens()}")
        logging.info(f"Total tokens in total: {self.get_total_tokens()}")
        logging.info(f"Ratio of gutenberg tokens to total tokens: {self.gutenberg_tokens / self.total_tokens}")


if __name__ == "__main__":
    config = DataConfig()
    print("Total number of sequences in Gutenberg:", config.get_gutenberg_sequences())
    print("Total number of tokens in Gutenberg:", config.get_gutenberg_tokens())
    print("Ratio of gutenberg tokens to total tokens:", config.gutenberg_tokens / config.total_tokens)