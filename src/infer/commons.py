import os
import json
from tqdm import tqdm
import torch
import numpy as np
import random
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn

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

    for sequence in batch['input_ids']:
        # No need to tokenize again if sequences are already tokenized (input_ids)
        # Adjust slicing based on offset, prefix, and suffix lengths
        prefix_suffix = sequence[_offset:_offset + _prefix_len + _suffix_len]
        prefix_suffix_list.append(prefix_suffix)

    return {
        'prefix_suffix': prefix_suffix_list,
    }

def calc_generation_nll(generated_sequences, scores):
    """
    Calculate negative log likelihood for each generated sequence.
    
    Args:
        generated_sequences (torch.Tensor): Token sequences [batch_size, seq_length]
        scores (List[torch.Tensor]): List of score tensors, each [batch_size, vocab_size], length of scores tensor is equal to seq_length
    
    Returns:
        tuple: (seq_nlls_mean, seq_nlls_std) - Mean and std of NLL per sequence
    """
    suffix = generated_sequences[:, -len(scores):]
    # assert suffix.shape[1] == generated_sequences.shape[1] // 2, f"Prefix suffix length mismatch: {suffix.shape[1]}"

    token_nlls = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for step, logits in enumerate(scores):
        step_nll = criterion(logits, suffix[:, step]) # Comparing batch_size x vocab_size with batch_size x 1, output is 
        token_nlls.append(step_nll)

        # Clear GPU memory
        del step_nll
        del logits
        if step % 5 == 0:  # Periodic memory cleanup
            torch.cuda.empty_cache()

    token_nlls = torch.stack(token_nlls, dim=-1) # shape: [batch_size, seq_length]
    assert token_nlls.min() >= 0, f"Negative NLL found: {token_nlls.min()}"
    return token_nlls, token_nlls.mean(dim=-1), token_nlls.std(dim=-1)


def run(model, dataset, prefix_length, suffix_length, batch_size, inference_dir, policy, seed):
    """Run distributed inference across multiple nodes and GPUs."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  # Global rank across all nodes
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

    # Set same seed for all ranks
    set_seed(seed)

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    model = model.to(local_rank)

    # Setup distributed sampling
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda batch: batch)
    
    # Create inference directory for this repetition
    inference_dir.mkdir(parents=True, exist_ok=True)
    output_file = inference_dir / f"rank{rank}.jsonl"

    generation_configs = {
        "greedy": {
            "num_beams": 1,
            "do_sample": False
        },
        "nucleus": {
            "num_beams": 1,
            "do_sample": True,
            "temperature": 1,
            "top_p": 0.3
        }
    }

    # Process batches
    with open(output_file, "w") as jsonl_file:
        for batch in tqdm(dataloader, 
                         desc=f"Generating Suffix (Rank {rank}/{world_size-1})", 
                         unit='batch', 
                         ncols=100, 
                         disable=rank != 0):
            
            # Clear cache before processing new batch
            torch.cuda.empty_cache()
            
            batch_tensor = torch.tensor(batch, device=local_rank)

            # Prepend <BoS> token
            # Prepend multiple tokens including <BoS>
            prepend_tokens = torch.tensor([128000], device=batch_tensor.device) 
            input_with_bos = torch.cat([
                prepend_tokens.repeat(batch_tensor.shape[0], 1),
                batch_tensor[:, :prefix_length]
            ], dim=1)

            assert input_with_bos.shape[1] == prefix_length + len(prepend_tokens), f"Input shape mismatch: {input_with_bos.shape}"
            assert batch_tensor.shape[1] == prefix_length + suffix_length, f"Batch shape mismatch: {batch_tensor.shape}"

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_with_bos,
                    max_new_tokens=suffix_length,
                    min_new_tokens=suffix_length,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generation_configs[policy]
                )

            sequences = outputs.sequences
            seq_nlls, seq_nlls_mean, seq_nlls_std = calc_generation_nll(sequences, outputs.scores)

            # Validate shapes
            assert sequences.shape[1] == len(prepend_tokens) + prefix_length + suffix_length, f"Output shape mismatch: {sequences.shape}"

            # Process and write batch results
            prefixes           = batch_tensor[:, :prefix_length].cpu().tolist() 
            true_suffixes      = batch_tensor[:, prefix_length:].cpu().tolist()
            generated_suffixes = sequences[:, prefix_length+len(prepend_tokens):].cpu().tolist() # Skip prepend BOS token

            nlls      = seq_nlls.cpu().tolist()
            nll_means = seq_nlls_mean.cpu().tolist()
            nll_stds  = seq_nlls_std.cpu().tolist()

            # Clear GPU tensors immediately after use
            del batch_tensor, sequences, outputs, input_with_bos
            del seq_nlls, seq_nlls_mean, seq_nlls_std

            # Write results directly without storing in memory
            for p, t, g, nll, nll_m, nll_s in zip(prefixes, true_suffixes, generated_suffixes, nlls, nll_means, nll_stds):
                json.dump({
                    "prefix": p,
                    "true_suffix": t,
                    "generated_suffix": g,
                    "nll": nll,
                    "nll_mean": nll_m,
                    "nll_std": nll_s
                }, jsonl_file)
                jsonl_file.write('\n')
                jsonl_file.flush()

            # Clear CPU lists after writing
            del prefixes, true_suffixes, generated_suffixes, nlls, nll_means, nll_stds
            torch.cuda.empty_cache()
    
    # Synchronize all processes
    dist.barrier()