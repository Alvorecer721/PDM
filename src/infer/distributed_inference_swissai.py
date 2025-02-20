from transformers import AutoModelForCausalLM, AutoConfig
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
from datasets import load_dataset

from distributed_inference_sparse import run
from distributed_inference import batch_processing_gutenberg
from utils import set_seed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert SwissAI Megatron checkpoint and run inference on Gutenberg dataset')
    parser.add_argument('--experiment-path', type=str, required=True, 
                      help='Path to experiment directory')
    parser.add_argument('--data-folder', type=str,
                      default='/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg',
                      help='Path to Gutenberg dataset folder')
    parser.add_argument('--repetitions', type=str, required=True,
                      help='Repetition choices, e.g. 128,256,512')
    
    parser.add_argument('--offset', type=int, default=0,
                      help='Offset for text processing, should always be larger then goldfish H')
    parser.add_argument('--prefix-length', type=int, default=500,
                      help='Length of prefix sequence')
    parser.add_argument('--suffix-length', type=int, default=500,
                      help='Length of suffix sequence')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for inference')
    parser.add_argument('--num-proc', type=int, default=20,
                      help='Number of processes for dataset mapping')
    parser.add_argument('--gen-policy', type=str, default='greedy',
                      help='Generation policy for inference, options: greedy, nucleus')
    
    args = parser.parse_args()

    # Set global seed for reproducibility
    set_seed(42)

    # Find the iteration directory dynamically
    model_path = Path(args.experiment_path) / "HF"

    # Create output directory
    output_dir = Path(args.experiment_path) / "inference"

    # Parse repetitions from command line
    repetitions = set([int(rep) for rep in args.repetitions.split(',')])
    
    # Create output directory with experiment parameters
    output_path = output_dir / f"offset_{args.offset}_prefix_{args.prefix_length}_suffix_{args.suffix_length}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all relevant data files matching requested repetitions
    data_folder = Path(args.data_folder)
    paths = sorted(
        (path for path in data_folder.glob("rep_*_token.jsonl")
        if int(path.stem.split('_')[1]) in repetitions),
        key=lambda path: int(path.stem.split('_')[1])
    )

    # Process each repetition
    for path in paths:
        rep = int(path.stem.split('_')[1])
        
        # Check if inference already exists for this repetition
        inference_dir = output_path / f"rep_{rep}_{args.gen_policy}"
        if inference_dir.exists():
            print(f"Skipping repetition {rep} - already processed")
            continue
            
        print(f"\nProcessing repetition {rep}")
        
        # Load and process dataset
        bucket = load_dataset("json", data_files=str(path), split='train')
        bucket = bucket.map(
            batch_processing_gutenberg,
            batched=True,
            desc=f"Generating prefix-suffix pairs for rep {rep}",
            num_proc=args.num_proc,
            fn_kwargs={
                '_prefix_len': args.prefix_length,
                '_suffix_len': args.suffix_length,
                '_offset': args.offset
            }
        )['prefix_suffix']
        
        # Validate sequence lengths
        assert len(bucket[0]) == args.prefix_length + args.suffix_length, \
            f"Sequence length mismatch for rep {rep}: got {len(bucket[0])}, expected {args.prefix_length + args.suffix_length}"
        
        print(f"Processing {len(bucket)} samples for repetition {rep}")
        
        # Run distributed inference for this repetition
        run(
            model=model,
            dataset=bucket,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            batch_size=args.batch_size,
            inference_dir=inference_dir,
            policy=args.gen_policy
        )
        
        print(f"Completed processing repetition {rep}")
        
        # Clear any cached tensors
        torch.cuda.empty_cache()
    
    print(f"\nAll repetitions processed. Results saved to: {output_path}")