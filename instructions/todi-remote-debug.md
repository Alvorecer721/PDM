1. Install VS Code CLI on Todi login node (one time operation)

   ```bash
   curl -L -o code.tar.gz "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
   tar -xf code.tar.gz
   ```

2.  Start a VS Code Server tunnel in a computing node

    ```bash
    srun --time 00:30:00 -A a06 --environment /store/swissai/a06/.NeMo/container/nemo.toml --partition debug --container-mounts=./code:/code --pty /code tunnel --accept-server-license-terms
    ```

3. Create your debug `launch.json` file, e.g.:

   ```json
   {
       // Use IntelliSense to learn about possible attributes.
       // Hover to view descriptions of existing attributes.
       // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
       "version": "0.2.0",
       "configurations": 
       [
           {
               "name": "Python Debugger: Figure out data mixing",
               "type": "debugpy",
               "request": "launch",
               "console": "integratedTerminal",
               "python": "/usr/bin/python3",
               "justMyCode": false,
               "env": {
                   "TRANSFORMERS_OFFLINE": "0",
                   "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                   "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                   "CUDA_VISIBLE_DEVICES": "0",
                   "PYTHONPATH": "/mloscratch/homes/yixuan/NeMo:/opt/Megatron-LM"
            },
               },
               "module": "torch.distributed.run",
               "args": [
                   "/users/xyixuan/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py",
                   "--config-path", "/users/xyixuan/store/.NeMo/Goldfish_Llama3/PDM/config",
                   "--config-name", "goldfish_debug.yaml",
                   "llama_param_size=1.5B",
                   "run.name=llama_1.5B_Goldfish_K_54_H_13_GBS_120_EPOCH_76",
                   "run.results_dir=/store/a06/.NeMo/Goldfish_Llama3/1.5B/Goldfish_K_54_H_13/GBS_120_EPOCH_76",
                   "model.global_batch_size=120",
                   "model.data.goldfish_loss=true",
                   "model.data.goldfish_h=13",
                   "model.data.goldfish_k=54",
                   "model.gc_interval=100",
                   "exp_manager.checkpoint_callback_params.every_n_train_steps=75",
                   "trainer.max_steps=5700",
               ]
           }
       ]
   }
   ```
4. Save and Export Breakpoints:
    * Install the [breakpointIO](https://github.com/redspart/breakpoint-io) extension in VS Code.
    * Press `Cmd + Shift + P` (Mac) or `Ctrl + Shift + P` (Windows/Linux) to open command palette, and type `breakpointio-export` to save your breakpoints to [breakpoints.json](../debug/rcp/breakpoints.json) under `.vscode` folder on your remote cluster. 
    * To restore the breakpoints later, open the command palette again and run `breakpointio-import`.

### Troubleshooting
---
1. Since [NeMo](https://github.com/TJ-Solergibert/NeMo) imports [Megatron-LM](https://github.com/TJ-Solergibert/Megatron-LM/tree/goldfish), you may find that your debugger steps into the Megatron-LM framework. In subsequent debugging sessions—when you’ve run `breakpointio-import` you might see a "The editor could not be opened because the file was not found" error if the Megatron-LM source files are inaccessible. To fix this issue:
    
    * Create a symbolic link to Megatron-LM under your workspace directory:
        ```bash
        ln -s /opt/Megatron-LM Megatron-LM
        ```
    *  Then update your PYTHONPATH in `launch.json` to use the symbolic link instead:
        ```json
        "PYTHONPATH": "/mloscratch/homes/yixuan/NeMo:/mloscratch/homes/yixuan/Megatron-LM"
        ```