from src.infer.distributed_inference import load_model
from transformers import AutoConfig
import shutil
import os
import argparse 
import glob

def get_checkpoint_path(expr_path):
   # Construct path to NeMo2HF directory
   nemo2hf_path = os.path.join(expr_path, 'results', 'NeMo2HF')
   
   # Find checkpoint files (assuming they end with .bin)
   ckpts = glob.glob(os.path.join(nemo2hf_path, '*.bin'))
   
   if not ckpts:
       raise ValueError(f"No checkpoint found in {nemo2hf_path}")
   if len(ckpts) > 1:
       raise ValueError(f"Multiple checkpoints found in {nemo2hf_path}. Please specify one.")
       
   return ckpts[0]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert checkpoint to HuggingFace format')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('--expr', type=str, required=True,
                        help='Path to experiment directory')
    args = parser.parse_args()

    # Check if HF directory already exists
    hf_dir = os.path.join(args.expr, 'results', 'HF')
    if os.path.exists(hf_dir):
        print(f"HF directory already exists at {hf_dir}, skipping conversion")
        return

    # Get checkpoint path automatically
    ckpt_path = get_checkpoint_path(args.expr)
    print(f"Found checkpoint: {ckpt_path}")

    # Create HF directory at same level as NeMo2HF
    ckpt_dir = os.path.dirname(ckpt_path)  # Get NeMo2HF directory
    parent_dir = os.path.dirname(ckpt_dir)  # Get parent directory
    hf_dir = os.path.join(parent_dir, 'HF')  # Create HF directory path
    os.makedirs(hf_dir, exist_ok=True)

    # Copy config
    shutil.copy(args.config, os.path.join(hf_dir, "config.json"))

    # Load config and model
    config = AutoConfig.from_pretrained(args.config)
    model = load_model(
        config=config,
        model_path=ckpt_path
    )

    # Save in HF format
    model.save_pretrained(hf_dir)
    print(f"Model saved to {hf_dir}")

if __name__ == "__main__":
    main()
