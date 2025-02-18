from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig
import os
from collections import deque
import json
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from distributed_inference import (
    batch_processing_gutenberg,
    load_model
)
from utils import set_seed


from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

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


def run(model, dataset, prefix_length, suffix_length, batch_size, inference_dir, policy):
    """Run distributed inference across multiple nodes and GPUs."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  # Global rank across all nodes
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

    # Set same seed for all ranks
    set_seed(42)

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
            prepend_tokens = torch.tensor([128000], device=batch_tensor.device) # , 79689, 4477, 25
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLaMA inference on Gutenberg dataset')

    # Required arguments
    parser.add_argument('--llama-config', type=str, default='/capstor/users/cscs/xyixuan/PDM/config/llama3_1.5B_config.json',
                      help='Path to the LLaMA model configuration')
    parser.add_argument('--experiment-path', type=str, 
                      required=True, 
                      help='Path to experiment directory')
    parser.add_argument('--data-folder', type=str,
                      default='/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg',
                      help='Path to Gutenberg dataset folder')
    parser.add_argument('--repetitions', type=str, required=True,
                      help='Repetition choices, e.g. 128,256,512')

    # Optional inference parameters
    parser.add_argument('--offset', type=int, default=0,
                      help='Offset for text processing, should always be larger then goldfish H')
    parser.add_argument('--prefix-length', type=int, default=500,
                      help='Length of prefix sequence')
    parser.add_argument('--suffix-length', type=int, default=500,
                      help='Length of suffix sequence')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for inference')
    parser.add_argument('--num-proc', type=int, default=20,
                      help='Number of processes for dataset mapping')
    parser.add_argument('--gen-policy', type=str, default='greedy',
                      help='Generation policy for inference, options: greedy, nucleus')

    args = parser.parse_args()

    # Set global seed before everything
    set_seed(42)

    llama_config = '/capstor/users/cscs/xyixuan/PDM/config/llama3_1.5B_config.json'
    experiment_path = Path(args.experiment_path)
    
    data_folder = Path("/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg")
    data_folder = Path(args.data_folder)
    
    config = AutoConfig.from_pretrained(args.llama_config)
    model_path = next(experiment_path.glob('results/NeMo2HF/step=*.bin')) # only the last checkpoint is converted
    model = load_model(config, model_path=str(model_path))

    output_path = experiment_path / 'inference' / f"offset_{args.offset}_prefix_{args.prefix_length}_suffix_{args.suffix_length}"
    output_path.mkdir(parents=True, exist_ok=True)

    policy = args.gen_policy
    repetitions = set([int(rep) for rep in args.repetitions.split(',')])

    paths = sorted(
        (path for path in data_folder.glob("rep_*_token.jsonl")
        if int(path.stem.split('_')[1]) in repetitions),
        key=lambda path: int(path.stem.split('_')[1])
    )

    for path in paths:
        rep = int(path.stem.split('_')[1])

        # Check if corresponding inference file exists
        inference_dir = output_path / f"rep_{rep}_{policy}"
        if inference_dir.exists():
            logging.info(f"Skipping repetition {rep} - already infered")
            continue

        bucket = load_dataset("json", data_files=str(path), split='train')
        bucket = bucket.map(
            batch_processing_gutenberg,
            batched=True,
            desc="Generating prefix and suffix pairs",
            num_proc=args.num_proc,
            fn_kwargs={
                '_prefix_len': args.prefix_length,
                '_suffix_len': args.suffix_length, 
                '_offset': args.offset
            }
        )['prefix_suffix']

        assert len(bucket[0]) == args.prefix_length + args.suffix_length, \
            f"Sequence length mismatch: got {len(bucket[0])}, expected {args.prefix_length + args.suffix_length}"

        logging.info(f"Processing repetition {rep} with {len(bucket)} samples")

        run(model, bucket, args.prefix_length, args.suffix_length, 
            args.batch_size, inference_dir, policy)