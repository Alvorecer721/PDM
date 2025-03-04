import os
import json
from tqdm import tqdm
import torch
import numpy as np
import random
import logging
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
from transformers import AutoModelForCausalLM
from datasets import load_dataset


import sys
# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verbatim_eval.LCS import find_longest_common_substrings
from verbatim_eval.my_rouge import compute_rouge_l_2d, _compute_dp_matrix_2d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def is_rank_0():
    """Helper function to check if current process is rank 0"""
    # Check if we're in a distributed environment
    if dist.is_initialized():
        return dist.get_rank() == 0
    # If not distributed, we're on rank 0
    return True



def setup_distributed(seed):
    """
    Set up distributed training environment.
    
    Args:
        seed (int): Random seed
        
    Returns:
        tuple: (local_rank, rank, world_size)
    """
    # Set random seed
    set_seed(seed)
    
    # Get distributed environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  # Global rank across all nodes
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

    # Initialize process group if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


########################################################################################################################
########################################## PROCESS DATASET #############################################################
########################################################################################################################


def batch_processing_gutenberg(batch, _prefix_len, _offset, _suffix_len=None):
    """
    Tokenize sequences from a batch of articles between specified character positions.

    Args:
        batch (dict): Batch of data containing the 'text' field.
        _tokenizer (AutoTokenizer): The tokenizer used for tokenization.
        _prefix_len (int): Length of the prefix to extract.
        _suffix_len (int, optional): Length of the suffix to extract. If None, defaults to prefix length.
        _offset (int, optional): Starting position for token slicing. Default is 0.

    Returns:
        dict: Dictionary containing 'prefix_list' and 'suffix_list'.
    """
    if _suffix_len is None:
        _suffix_len = _prefix_len

    prefix_suffix_list = []

    for sequence in batch["input_ids"]:
        # No need to tokenize again if sequences are already tokenized (input_ids)
        # Adjust slicing based on offset, prefix, and suffix lengths
        prefix_suffix = sequence[_offset : _offset + _prefix_len + _suffix_len]
        prefix_suffix_list.append(prefix_suffix)

    return {
        "prefix_suffix": prefix_suffix_list,
    }


def process_dataset(
    data_path, batch_processing_fn, prefix_length, suffix_length, offset, num_proc
):
    """
    Load and process a dataset for model inference.

    This function loads data from a JSON file and applies a batched processing function
    to generate prefix-suffix pairs for inference tasks.

    Args:
        data_path (str or Path): Path to the JSON file(s) containing the dataset.
            Can be a single file or a pattern matching multiple files.
        batch_processing_fn (callable): Function to process the dataset in batches.
            Must accept batched data and keyword arguments for prefix_len, suffix_len, and offset.
        prefix_length (int): Length of the prefix sequence to generate.
        suffix_length (int): Length of the suffix sequence to generate.
        offset (int): Offset position to start processing text.
        num_proc (int): Number of processes to use for parallel dataset mapping.

    Returns:
        Dataset: A processed dataset containing prefix-suffix pairs ready for inference.

    Raises:
        ValueError: If the dataset cannot be loaded or processed.
    """
    logger.info(f"Processing dataset from {data_path}")

    # Load dataset (works with both file paths and file patterns)
    dataset = load_dataset(
        "json", 
        data_files=str(data_path), 
        split="train", 
        cache_dir="/iopsstor/scratch/cscs/xyixuan/cache"
    )

    processed_dataset = dataset.map(
        batch_processing_fn,
        batched=True,
        desc="Generating prefix and suffix pairs",
        num_proc=num_proc,
        fn_kwargs={
            "_prefix_len": prefix_length,
            "_suffix_len": suffix_length,
            "_offset": offset,
        },
    )["prefix_suffix"]

    logger.info(f"Processed {len(processed_dataset)} samples")
    return processed_dataset


########################################################################################################################
##################################################### PATH #############################################################
########################################################################################################################


def setup_output_directories(experiment_path, offset, prefix_length, suffix_length):
    """
    Set up the output directories for inference results.

    Args:
        experiment_path (Path): Base path for the experiment.
        offset (int): Offset value used in text processing.
        prefix_length (int): Length of prefix used.
        suffix_length (int): Length of suffix used.

    Returns:
        Path: Path to the output directory.
    """
    output_dir = experiment_path / "inference"
    output_path = (
        output_dir / f"offset_{offset}_prefix_{prefix_length}_suffix_{suffix_length}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def get_inference_dir(output_path, rep, gen_policy):
    """
    Get the inference directory for a specific repetition and policy.

    Args:
        output_path (Path): Base output path.
        rep (int): Repetition number.
        gen_policy (str): Generation policy name.

    Returns:
        Path: Path to the inference directory.
    """
    inference_dir = output_path / f"rep_{rep}_{gen_policy}"
    return inference_dir


########################################################################################################################
#################################################### TEXT METRICS ######################################################
########################################################################################################################


def calc_generation_nll(generated_sequences, scores):
    """
    Calculate negative log likelihood for each generated sequence.

    Args:
        generated_sequences (torch.Tensor): Token sequences [batch_size, seq_length]
        scores (List[torch.Tensor]): List of score tensors, each [batch_size, vocab_size], length of scores tensor is equal to seq_length

    Returns:
        tuple: (seq_nlls_mean, seq_nlls_std) - Mean and std of NLL per sequence
    """
    suffix = generated_sequences[:, -len(scores) :]

    token_nlls = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    for step, logits in enumerate(scores):
        step_nll = criterion(
            logits, suffix[:, step]
        )  # Comparing batch_size x vocab_size with batch_size x 1, output is
        token_nlls.append(step_nll)

        # Clear GPU memory
        del step_nll
        del logits
        if step % 5 == 0:  # Periodic memory cleanup
            torch.cuda.empty_cache()

    token_nlls = torch.stack(token_nlls, dim=-1)  # shape: [batch_size, seq_length]
    assert token_nlls.min() >= 0, f"Negative NLL found: {token_nlls.min()}"

    seq_nlls_mean = token_nlls.mean(dim=-1)  # Average NLL per sequence
    seq_nlls_std = token_nlls.std(dim=-1)    # Std of NLL per sequence
    
    # Calculate perplexity per sequence: exp(average NLL)
    perplexity = torch.exp(seq_nlls_mean)
    
    return token_nlls, seq_nlls_mean, seq_nlls_std, perplexity


def calculate_text_metrics(true_seq, gen_seq):
    """
    Calculate various text similarity metrics for a pair of sequences.

    Args:
        true_seq (list): True suffix sequence (token IDs)
        gen_seq (list): Generated suffix sequence (token IDs)

    Returns:
        dict: Dictionary with metrics (TTR_ref, TTR_gen, Rouge-L)
    """
    # Type-Token-Ratio
    ttr_ref = len(set(true_seq)) / len(true_seq) if true_seq else 0
    ttr_gen = len(set(gen_seq)) / len(gen_seq) if gen_seq else 0

    # Rouge-L
    dp_matrix = _compute_dp_matrix_2d(true_seq, gen_seq)
    rouge_l = compute_rouge_l_2d(dp_matrix)
    del dp_matrix  # Free memory

    return {
        "TTR_ref": ttr_ref,
        "TTR_gen": ttr_gen,
        "Rouge-L": rouge_l
    }

########################################################################################################################
################################################## INFERENCE ###########################################################
########################################################################################################################


def load_model(model_path):
    """Load the model from the specified path."""
    logger.info(f"Loading model from {model_path}")

    if not model_path.exists():
        raise ValueError(f"Model checkpoint not found at {model_path}")

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )


def run(
    model,
    dataset,
    prefix_length,
    suffix_length,
    batch_size,
    inference_dir,
    policy,
    seed,
):
    """Run distributed inference across multiple nodes and GPUs."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  # Global rank across all nodes
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes
    model.to(local_rank)

    # Set same seed for all ranks
    setup_distributed(seed)

    # Setup distributed sampling
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda batch: batch
    )

    # Create inference directory for this repetition
    inference_dir.mkdir(parents=True, exist_ok=True)
    output_file = inference_dir / f"rank{rank}.jsonl"

    generation_configs = {
        "greedy": {"num_beams": 1, "do_sample": False},
        "nucleus": {"num_beams": 1, "do_sample": True, "temperature": 1, "top_p": 0.3},
    }


    # Process batches
    with open(output_file, "w") as jsonl_file:
        for batch in tqdm(
            dataloader,
            desc=f"Generating Suffix (Rank {rank}/{world_size-1})",
            unit="batch",
            ncols=100,
            disable=rank != 0,
        ):

            # Clear cache before processing new batch
            torch.cuda.empty_cache()

            batch_tensor = torch.tensor(batch, device=local_rank)

            # Prepend <BoS> token
            # Prepend multiple tokens including <BoS>
            prepend_tokens = torch.tensor([128000], device=batch_tensor.device)
            input_with_bos = torch.cat(
                [
                    prepend_tokens.repeat(batch_tensor.shape[0], 1),
                    batch_tensor[:, :prefix_length],
                ],
                dim=1,
            )

            assert input_with_bos.shape[1] == prefix_length + len(
                prepend_tokens
            ), f"Input shape mismatch: {input_with_bos.shape}"
            assert (
                batch_tensor.shape[1] == prefix_length + suffix_length
            ), f"Batch shape mismatch: {batch_tensor.shape}"

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_with_bos,
                    max_new_tokens=suffix_length,
                    min_new_tokens=suffix_length,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generation_configs[policy],
                )

            sequences = outputs.sequences
            seq_nlls, seq_nlls_mean, seq_nlls_std, perplexity = calc_generation_nll(
                sequences, outputs.scores
            )

            # Validate shapes
            assert (
                sequences.shape[1]
                == len(prepend_tokens) + prefix_length + suffix_length
            ), f"Output shape mismatch: {sequences.shape}"

            # Process and write batch results
            prefixes = batch_tensor[:, :prefix_length].cpu().tolist()
            true_suffixes = batch_tensor[:, prefix_length:].cpu().tolist()
            generated_suffixes = (
                sequences[:, prefix_length + len(prepend_tokens) :].cpu().tolist()
            )  # Skip prepend BOS token

            lcs_result = find_longest_common_substrings(true_suffixes, generated_suffixes)
            lcs = lcs_result['max_length'].to_numpy()
            lcs_norm = lcs / suffix_length

            lcs = lcs.tolist()
            nlls = seq_nlls.cpu().tolist()
            nll_means = seq_nlls_mean.cpu().tolist()
            nll_stds = seq_nlls_std.cpu().tolist()
            perplexities = perplexity.cpu().tolist()

            # Clear GPU tensors immediately after use
            del batch_tensor, sequences, outputs, input_with_bos
            del seq_nlls, seq_nlls_mean, seq_nlls_std, perplexity

            # Write results directly without storing in memory
            for p, t, g, nll, nll_m, nll_s, lcs_n, ppl in zip(
                prefixes, true_suffixes, generated_suffixes, nlls, nll_means, nll_stds, lcs_norm, perplexities
            ):
                metrics = calculate_text_metrics(t,g)

                json.dump(
                    {
                        "prefix": p,
                        "true_suffix": t,
                        "generated_suffix": g,
                        "nll": nll,
                        "nll_mean": nll_m,
                        "nll_std": nll_s,
                        "perplexity": ppl,
                        "lcs_norm": lcs_n,
                        **metrics
                    },
                    jsonl_file,
                )
                jsonl_file.write("\n")
                jsonl_file.flush()

            # Clear CPU lists after writing
            del prefixes, true_suffixes, generated_suffixes
            del nlls, nll_means, nll_stds, perplexities
            del lcs, lcs_norm, lcs_result
            torch.cuda.empty_cache()

    # Synchronize all processes
    dist.barrier()
