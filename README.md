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

### Plots

To generate plots there are two options:

1. Compare multiple models on common benchmarks (bar plot)
   - What it does: Loads multiple lm-eval results JSON files and produces a grouped bar chart with accuracy (and stderr) per benchmark, one color per model.
   - How to run:
     ```bash
     python -m lm_eval.plot_eval_results --output-path /path/to/plots/llm_comparison.pdf [--use-latex-text-renderer]
     ```
     Notes:
     - Edit the script to list the result JSON files you want to compare.
     - The optional --use-latex-text-renderer flag uses a local LaTeX installation for text rendering (if available).

2. Track a single model over training/checkpoints (line plot)
   - What it does: Recursively collects lm-eval results JSON files from one or more result folders and plots accuracy with uncertainty over training progress (consumed tokens) for several benchmarks.
   - How to run:
     ```bash
     python -m lm_eval.plot_eval_results_over_iter --output-path /path/to/plots/training_progress.pdf [--use-latex-text-renderer]
     ```
     Notes:
     - Edit the script to list the folders containing your evaluation results; JSON files inside will be picked up automatically.
     - The horizontal axis reflects consumed tokens computed from checkpoint metadata and run settings embedded in paths; adjust assumptions in the script if your setup differs.
