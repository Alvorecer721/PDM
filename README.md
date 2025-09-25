# PDM
Master Thesis at EPFL &amp; ETHz &amp; Swiss AI Initiative

## Usage

To **convert Megatron checkpoints into Hugging Face checkpoints** and run benchmark evaluations, follow these steps:

1. Navigate to the `submissions` folder.  
2. Update the output paths in the `SBATCH` script to point to a directory accessible to you.  
3. Submit the job with a command like:

   ```bash
   sbatch submit-lm-eval.slurm /iopsstor/scratch/cscs/nirmiger/Megatron-LM/logs/Meg-Runs/image-extension/llama3-3b-15n-8192sl-120gbsz-0.8i-0.2t-long-run
    ```

This process will first convert the checkpoint and then run model evaluations using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

To customize the evaluation (e.g., tokenizer model, selected benchmarks, etc.), edit:

```bash
scripts/swissai_megatron/convert-and-lm-eval.sh
```
Review the options in the final command of the script to adjust evaluation settings.

The `convert-and-lm-eval.sh` script will:  
- Store evaluation results in the `results/lm_eval` folder.  
- Save the converted Hugging Face model in the experiment directory of `Megatron-LM` associated with the `submit-lm-eval.slurm` job. 

👉 If instead of running `submit-lm-eval.slurm` you run `submit-mllm-eval.slurm`, the same process occurs, but evaluations are performed using the [lmms-eval](https://github.com/nirmiger/lmms-eval.git) repository for **vision-language benchmarks**.  


