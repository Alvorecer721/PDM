# distributed_inference_dense.py
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
    parser = argparse.ArgumentParser(description='Convert SwissAI Megatron checkpoint and run inference on Gutenberg dataset')
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
    experiment_path = Path(args.experiment_path)

    # Set global seed for reproducibility
    set_seed(args.seed)

    # Load model
    experiment_path = Path(args.experiment_path)
    model_path = experiment_path / "HF"
    model = load_model(model_path)
    
    # Set up output directories
    output_path = setup_output_directories(
        experiment_path,
        args.offset,
        args.prefix_length,
        args.suffix_length
    )

    # Parse repetitions from command line
    repetitions = set([int(rep) for rep in args.repetitions.split(',')])
    
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
        inference_dir = get_inference_dir(output_path, rep, args.gen_policy)
        if inference_dir.exists():
            logger.info(f"Skipping repetition {rep} - already processed")
            continue
            
        logger.info(f"\nProcessing repetition {rep}")
        
        # Load and process dataset
        # Process dataset for this repetition
        bucket = process_dataset(
            path,
            batch_processing_gutenberg,
            args.prefix_length,
            args.suffix_length,
            args.offset,
            args.num_proc
        )
        
        # Validate sequence lengths
        assert len(bucket[0]) == args.prefix_length + args.suffix_length, \
            f"Sequence length mismatch for rep {rep}: got {len(bucket[0])}, expected {args.prefix_length + args.suffix_length}"
        
        logger.info(f"Processing {len(bucket)} samples for repetition {rep}")
        
        # Run distributed inference for this repetition
        run(
            model=model,
            dataset=bucket,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            batch_size=args.batch_size,
            inference_dir=inference_dir,
            policy=args.gen_policy,
            seed=args.seed,
        )
        
        print(f"Completed processing repetition {rep}")
        
        # Clear any cached tensors
        torch.cuda.empty_cache()
    
    print(f"\nAll repetitions processed. Results saved to: {output_path}")