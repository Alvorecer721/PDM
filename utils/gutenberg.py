import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import random
from collections import Counter
import os


def batch_tokenize_gutenberg(batch, _tokenizer, char_pos_start, char_pos_end):
    """
    Tokenize sequences from a batch of articles between specified character positions.

    Args:
        batch (dict): Batch of data containing the 'text' field.
        _tokenizer (AutoTokenizer): The tokenizer used for tokenization.
        char_pos_start (int, optional): Starting character position. Defaults to 10_000.
        char_pos_end (int, optional): Ending character position. Defaults to 70_000.

    Returns:
        dict: Dictionary containing input_ids and sequence lengths.
    """
    input_ids_list = []
    seq_length_list = []

    for sequence in batch["text"]:
        # Slice and tokenize the sequence
        input_ids = _tokenizer(
            sequence[char_pos_start:char_pos_end], truncation=False, padding=False
        ).input_ids
        input_ids_list.append(input_ids)
        seq_length_list.append(len(input_ids))

    return {"input_ids": input_ids_list, "seq_length": seq_length_list}


def select_tokens_from_random_offset(batch, _tokenizer, num_tokens, seed=42):
    """
    Select a sequence of tokens with random offset from each batch, detokenize them, and return the results.

    Args:
        batch (dict): Batch of data containing 'input_ids'.
        _tokenizer (AutoTokenizer): The tokenizer used for detokenization.
        num_tokens (int): Number of tokens to extract.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing selected tokens and detokenized texts.
    """
    # Create a separate random number generator for reproducibility
    rng = random.Random(seed)

    selected_tokens = []
    detokenized_texts = []

    for input_ids in batch["input_ids"]:
        offset = rng.randint(0, len(input_ids) - num_tokens)

        # Select the specified number of tokens starting from the random offset
        selected_ids = input_ids[offset : offset + num_tokens]
        selected_tokens.append(selected_ids)

        # Detokenize the selected tokens to get the original text
        detokenized_text = _tokenizer.decode(selected_ids, skip_special_tokens=True)
        detokenized_texts.append(detokenized_text)

    return {"selected_tokens": selected_tokens, "detokenized_texts": detokenized_texts}


def remove_duplicate_ids(example, seen_ids):
    """Helper function to filter out duplicates based on ID."""
    if example["id"] in seen_ids:
        return False
    seen_ids.add(example["id"])
    return True


def main(args):
    # Load the dataset and tokenizer
    ds = load_dataset("manu/project_gutenberg", split="en")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Set the tokenizer's model max length
    tokenizer.model_max_length = 200_000

    # Filter and select the subset of articles
    seen_ids = set()
    subset = (
        ds.filter(lambda x: remove_duplicate_ids(x, seen_ids))
        .filter(lambda example: len(example["text"]) >= args.char_pos_end)
        .select(range(args.num_articles))
    )

    # Verify no duplicates
    id_counts = Counter(subset["id"])
    assert len(id_counts) == len(subset), "There are duplicate IDs in the dataset"

    # Use all available cpu processors
    num_proc = os.cpu_count()

    # Tokenize the selected subset of articles
    gutenberg = subset.map(
        batch_tokenize_gutenberg,
        batched=True,
        desc="Tokenizing Gutenberg English articles",
        num_proc=num_proc,
        fn_kwargs={
            "_tokenizer": tokenizer,
            "char_pos_start": args.char_pos_start,
            "char_pos_end": args.char_pos_end,
        },
    )

    # Apply random token selection and detokenization
    gutenberg_with_tokens = gutenberg.map(
        select_tokens_from_random_offset,
        batched=True,
        desc="Selecting tokens from random offset and detokenizing",
        num_proc=num_proc,
        fn_kwargs={
            "_tokenizer": tokenizer,
            "num_tokens": args.num_tokens,
            "seed": args.seed,
        },
    )

    # Save the results
    gutenberg_with_tokens.select_columns(
        ["selected_tokens", "detokenized_texts"]
    ).to_json(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Gutenberg dataset for token selection"
    )
    parser.add_argument(
        "--num-articles", type=int, default=10_000, help="Number of articles to process"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=9_000,
        help="Number of tokens to extract per article",
    )
    parser.add_argument(
        "--char-pos-start",
        type=int,
        default=10_000,
        help="Starting character position for tokenization",
    )
    parser.add_argument(
        "--char-pos-end",
        type=int,
        default=70_000,
        help="Ending character position for tokenization",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="gutenberg_en_8k.jsonl",
        help="Output file path",
    )

    args = parser.parse_args()
    main(args)
