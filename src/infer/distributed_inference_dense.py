import argparse
from pathlib import Path
import logging
import torch

from commons import (
    set_seed, run, batch_processing_gutenberg, process_dataset,
    setup_output_directories, get_inference_dir, load_model
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on multiple repetitions')
    parser.add_argument('--experiment-path', type=str, required=True, 
                      help='Path to experiment directory')
    parser.add_argument('--data-folder', type=str,
                      default='/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg',
                      help='Path to Gutenberg dataset folder')
    parser.add_argument('--repetitions', type=str, required=True,
                      help='Repetition choices, e.g. 128,256,512')
    
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset for text processing')
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed for all ranks')
    
    args = parser.parse_args()

    # Set global seed for reproducibility
    set_seed(args.seed)

    # Find the iteration directory dynamically
    experiment_path = Path(args.experiment_path)
    model_path = experiment_path / "HF" / f"iter_{args.num_epoch*args.iterations_per_epoch:07d}"

    # Load model
    model = load_model(model_path)
    
    # Set up output directories
    output_path = setup_output_directories(
        experiment_path,
        args.offset,
        args.prefix_length,
        args.suffix_length
    )

    inference_dir = get_inference_dir(output_path, args.num_epoch, args.gen_policy)
    
    # Process dataset
    dataset = process_dataset(
        args.data_path,
        batch_processing_gutenberg,
        args.prefix_length,
        args.suffix_length,
        args.offset,
        args.num_proc
    )
    
    # Run inference
    logger.info("Starting inference")
    run(
        model=model,
        dataset=dataset,
        prefix_length=args.prefix_length,
        suffix_length=args.suffix_length,
        batch_size=args.batch_size,
        inference_dir=inference_dir,
        policy=args.gen_policy,
        seed=args.seed,
    )
        
    print(f"Completed inference")
    
    # Clear any cached tensors
    torch.cuda.empty_cache()