from pathlib import Path
import logging
import argparse
from transformers import AutoConfig
from distributed_inference import (
    batch_processing_gutenberg,
    load_model
)
from commons import run, set_seed


from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


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
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed for all ranks')
    
    args = parser.parse_args()

    # Set global seed before everything
    set_seed(args.seed)

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