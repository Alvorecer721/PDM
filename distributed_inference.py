import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import time
import json
from collections import deque
import argparse

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from datasets import load_dataset


def batch_processing_gutenberg(batch, _prefix_len, _suffix_len=None, _offset=0):
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


def load_model(config, model_path):
    """
    Load a model from a given path and convert to the specified precision.

    Args:
    - config: The configuration for the model.
    - model_path: Path to the model file (.bin) to load.
    - precision: The desired precision ('bf16', 'fp16', 'fp32'). Default is 'bf16'.

    Returns:
    - model: The loaded and precision-converted model.
    """
    # Load the model configuration
    model = AutoModelForCausalLM.from_config(config)

    # Load the exported model weights
    model_weights = torch.load(model_path, map_location='cpu')

    # Load the weights into the model
    model.load_state_dict(model_weights)

    return model.bfloat16()

    
def run(model, dataset, prefix_length, suffix_length, experiment_id, ckpt_id, batch_size):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
        
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    model = model.to(local_rank)

    # Create a DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda batch: batch)
    
    filename = f"{experiment_id}_{ckpt_id}_rank{local_rank}.jsonl"
    
    all_outputs = deque()
    all_batch_tensors = deque()
    
    for batch in tqdm(dataloader, desc=f"Generating Suffix (Rank {local_rank})", unit='batch', ncols=100):
        start_time = time.time()
        
        batch_tensor = torch.tensor(batch).to(local_rank)
        outputs = model.generate(
            input_ids=batch_tensor[:, :prefix_length],
            max_new_tokens=suffix_length,
            num_beams=1,
            do_sample=False
        )
        
        all_outputs.append(outputs.cpu().detach())
        all_batch_tensors.append(batch_tensor.cpu().detach())
        
        torch.cuda.empty_cache()
        
        end_time = time.time()
        time_per_seq = (end_time - start_time) / len(batch)
        if local_rank == 0:
            tqdm.write(f"Time per sequence: {time_per_seq:.3f} seconds")

    inference_folder = f"inference/{experiment_id}"
    os.makedirs(inference_folder, exist_ok=True)
    filename = os.path.join(inference_folder, f"{ckpt_id}_rank{local_rank}.jsonl")

    with open(filename, "w") as jsonl_file:
        for outputs, batch_tensor in tqdm(zip(all_outputs, all_batch_tensors), desc="Processing and writing data", total=len(all_outputs)):
            prefixes = batch_tensor[:, :prefix_length].tolist()
            true_suffixes = batch_tensor[:, prefix_length:].tolist()
            generated_suffixes = outputs[:, prefix_length:].tolist()
            
            batch_data = [
                {"prefix": p, "true_suffix": t, "generated_suffix": g} 
                for p, t, g in zip(prefixes, true_suffixes, generated_suffixes)
            ]
            
            for item in batch_data:
                json.dump(item, jsonl_file)
                jsonl_file.write('\n')
    
    dist.barrier()  # Wait for all processes to finish writing


if __name__ == "__main__":
    data_path      = '/store/swissai/a06/.NeMo/Goldfish_Llama3/data/gutenberg_en_8k_token.jsonl'
    llama_config   = '/store/swissai/a06/.NeMo/Goldfish_Llama3/hf_llama_config/llama3_1b_config.json'
    hf_login_token = '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'
    step           = 75
    consumed       = 9000
    llama_size     = '1.5B'
    prefix_length  = 500  # Specify your prefix length
    suffix_length  = 500  # Specify your suffix length
    batch_size     = 500
    experiment_id = f"llama_{llama_size}_Standard_GBS_120_EPOCH_75"

    parser = argparse.ArgumentParser(description='Run inference with specified parameters')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the tokenised jsonl file')
    parser.add_argument('--hf-token', type=str, required=True,
                        help='HuggingFace login token')
    parser.add_argument('--step', type=int, default=75,
                        help='Training step number')
    parser.add_argument('--consumed', type=int, default=9000,
                        help='Number of consumed samples')
    parser.add_argument('--llama-size', type=str, default='1.5B',
                        help='Size of the LLaMA model')
    parser.add_argument('--llama-config', type=str, required=True,
                        help='Path to the LLaMA model configuration')
    parser.add_argument('--prefix-length', type=int, default=500,
                        help='Length of the prefix sequence')
    parser.add_argument('--suffix-length', type=int, default=500,
                        help='Length of the suffix sequence')
    parser.add_argument('--seq-offset', type=int, default=0,
                        help='Starting position for token slicing')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing')
    parser.add_argument('--hf-model-path', type=str, default=None,
                        help='Path to the huggingface model checkpoint')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment ID, e.g. llama_1.5b_Standard_GBS_120_EPOCH_75')
    
    args = parser.parse_args()

    data_file      = args.data_path
    model_path     = args.hf_model_path
    llama_config   = args.llama_config
    hf_login_token = args.hf_token
    step           = args.step
    consumed       = args.consumed
    llama_size     = args.llama_size
    prefix_length  = args.prefix_length
    suffix_length  = args.suffix_length
    offset         = args.seq_offset
    batch_size     = args.batch_size
    experiment_id  = args.experiment
    ckpt_id       = f"step={step}-consumed={consumed}"
    

    login(token=hf_login_token)
    
    world_size = int(os.environ["WORLD_SIZE"])  # Number of GPUs
    config = AutoConfig.from_pretrained(llama_config)

    
    # Load the model and tokenizer 
    if not model_path:
        model_path = f"/store/swissai/a06/.NeMo/Goldfish_Llama3/{llama_size}/{experiment_id}/results/NeMo2HF/{ckpt_id}.bin"
    model = load_model(config, model_path)
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    # Load and process data
    gutenberg = load_dataset("json", data_files=data_file)
    dataset = gutenberg['train'].map(
        batch_processing_gutenberg,
        batched=True,
        desc="Generating prefix and suffix pairs",
        num_proc=world_size,
        fn_kwargs={
            '_prefix_len': prefix_length,
            '_suffix_len': suffix_length, 
            '_offset': offset
        }
    )['prefix_suffix']
    
    run(model, dataset, prefix_length, suffix_length, experiment_id, ckpt_id, batch_size)    