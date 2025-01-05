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

from distributed_inference import (
    batch_processing_gutenberg,
    load_model
)

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


def run(model, dataset, prefix_length, suffix_length, batch_size, inference_dir):
    """Run distributed inference across multiple nodes and GPUs."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])  # Global rank across all nodes
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    model = model.to(local_rank)

    # Setup distributed sampling
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda batch: batch)
    
    all_outputs = deque()
    all_batch_tensors = deque()

    # Create inference directory for this repetition
    inference_dir.mkdir(parents=True, exist_ok=True)
    output = inference_dir / f"rank{rank}.jsonl"

    for batch in tqdm(dataloader, desc=f"Generating Suffix (Rank {rank}/{world_size-1})", unit='batch', ncols=100, disable=rank != 0): 
        batch_tensor = torch.tensor(batch).to(local_rank)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_tensor[:, :prefix_length],
                max_new_tokens=suffix_length,
                num_beams=1,
                do_sample=False
            )

        all_outputs.append(outputs.cpu().detach())
        all_batch_tensors.append(batch_tensor.cpu().detach())
        
        torch.cuda.empty_cache()

    # Save results for this rank
    output = inference_dir / f"rank{rank}.jsonl"
    
    with open(output, "w") as jsonl_file:
        for outputs, batch_tensor in tqdm(
            zip(all_outputs, all_batch_tensors), 
            desc="Processing and writing", 
            total=len(all_outputs),
            disable=rank != 0
        ):
            prefixes = batch_tensor[:, :prefix_length].tolist()
            true_suffixes = batch_tensor[:, prefix_length:].tolist()
            generated_suffixes = outputs[:, prefix_length:].tolist()
            
            batch_data = [
                {
                    "prefix": p, 
                    "true_suffix": t, 
                    "generated_suffix": g
                } 
                for p, t, g in zip(prefixes, true_suffixes, generated_suffixes)
            ]
            
            for item in batch_data:
                formatted_line = (
                    "{\n"
                    f'    "prefix":            {json.dumps(item["prefix"])},\n'
                    f'    "true_suffix":       {json.dumps(item["true_suffix"])},\n'
                    f'    "generated_suffix":  {json.dumps(item["generated_suffix"])}\n'
                    "}\n"
                )
                jsonl_file.write(formatted_line)
    
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

    # Optional inference parameters
    parser.add_argument('--offset', type=int, default=0,
                      help='Offset for text processing')
    parser.add_argument('--prefix-length', type=int, default=500,
                      help='Length of prefix sequence')
    parser.add_argument('--suffix-length', type=int, default=500,
                      help='Length of suffix sequence')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for inference')
    parser.add_argument('--num-proc', type=int, default=20,
                      help='Number of processes for dataset mapping')

    args = parser.parse_args()

    llama_config = '/capstor/users/cscs/xyixuan/PDM/config/llama3_1.5B_config.json'
    experiment_path = Path(args.experiment_path)
    
    data_folder = Path("/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg")
    data_folder = Path(args.data_folder)
    
    config = AutoConfig.from_pretrained(args.llama_config)
    model_path = next(experiment_path.glob('results/NeMo2HF/step=*.bin')) # only the last checkpoint is converted
    model = load_model(config, model_path=str(model_path))

    output_path = experiment_path / 'inference'
    output_path.mkdir(parents=True, exist_ok=True)

    for path in data_folder.glob("rep_*_token.jsonl"):
        rep = int(path.stem.split('_')[1])

        # Check if corresponding inference file exists
        inference_dir = output_path / f"rep_{rep}"
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

        logging.info(f"Processing repetition {rep} with {len(bucket)} samples")

        run(model, bucket, args.prefix_length, args.suffix_length, 
            args.batch_size, inference_dir)