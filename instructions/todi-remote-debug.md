1. Install VS Code CLI on Todi login node

   ```shell
   curl -L -o code.tar.gz "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
   tar -xf code.tar.gz
   ```

2.  Start a VS Code Server tunnel in a computing node

   ```shell
   srun --time 00:30:00 -A a06 --environment /store/swissai/a06/.NeMo/container/nemo.toml --partition debug --container-mounts=./code:/code --pty /code tunnel --accept-server-license-terms
   ```

3. Create your debug `launch.json` file, e.g.:

   ```json
   {
       // Use IntelliSense to learn about possible attributes.
       // Hover to view descriptions of existing attributes.
       // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
       "version": "0.2.0",
       "configurations": [
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

   





