"""
python /capstor/users/cscs/xyixuan/PDM/src/verbatim_eval/main.py \
    --exprs llama3-1b-15n-8192sl-60gbsz-standard \
    --mode sparse \
    --offsets 0 \
    --prefix-lengths 50 500 1000 2000 3000 \
    --suffix-lengths 500 \
    --repetitions 1 2 4 8 16 32 64 128
"""

import argparse
from pathlib import Path
from typing import Optional
from controlled_expr import (
    EvalConfig, 
    eval_expr, 
    get_repetitions
)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for experiments")
    parser.add_argument("--exprs", nargs="+", required=True, help="List of experiment names")
    parser.add_argument("--mode", choices=["dense", "sparse", "swaps"], default="dense", 
                        help="Mode for saving results")
    parser.add_argument("--base-path", type=str, 
                        default="/iopsstor/scratch/cscs/xyixuan/Megatron-LM-BKP/logs/Meg-Runs/",
                        help="Base path for experiments")
    parser.add_argument("--offsets", type=int, nargs="+", 
                        default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                        help="List of offsets")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", 
                        default=[500],
                        help="List of prefix lengths")
    parser.add_argument("--suffix-lengths", type=int, nargs="+", 
                        default=[500],
                        help="List of suffix lengths")
    parser.add_argument("--repetitions", type=int, nargs="+", default=None,
                        help="List of repetition frequencies; if omitted, will use all available repetitions.")
    
    args = parser.parse_args()
    
    # Construct the full base path using mode
    if args.mode == "dense":
        full_base_path = Path(args.base_path) / "DenseGutenberg"
    elif args.mode == "sparse": 
        full_base_path = Path(args.base_path) / "SparseGutenberg"
    elif args.mode == "swaps":
        full_base_path = Path(args.base_path) / "Offset-Effect"
    
    # Process each experiment
    for expr in args.exprs:
        infer_path = full_base_path / expr / "inference" / f"offset_{args.offsets[0]}_prefix_{args.prefix_lengths[0]}_suffix_{args.suffix_lengths[0]}"
        
        # Check if inference path exists
        if not infer_path.exists():
            print(f"Error: Inference path does not exist: {infer_path}")
            continue
            
        # Get repetitions
        repetitions: Optional[list[int]] = args.repetitions
        if repetitions is None:
            repetitions = get_repetitions(infer_path)
        print(f"Found repetitions for {expr}: {repetitions}")
        
        # Create evaluation config
        eval_config = EvalConfig(
            base_path=str(full_base_path),
            expr=expr,
            repetitions=repetitions,
            policy="greedy",
            offsets=args.offsets,
            prefix_lengths=args.prefix_lengths, 
            suffix_lengths=args.suffix_lengths
        )
        
        # Run evaluation
        result = eval_expr(eval_config)
        
        # Save results
        result.save(args.mode)


if __name__ == "__main__":
    main()