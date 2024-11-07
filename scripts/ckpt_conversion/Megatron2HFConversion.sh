#!/bin/bash

# Directories
experiment_dir="/users/xyixuan/store/a06/.NeMo/Goldfish_Llama3/1.5B/llama_1.5B_Goldfish_K_10_H_13_GBS_120_EPOCH_83" 
checkpoint_dir="$experiment_dir/results/checkpoints"
nemo_output_dir="$experiment_dir/results/Megatron2NeMo"
hf_output_dir="$experiment_dir/results/NeMo2HF"
hparams_file="$experiment_dir/results/hparams.yaml"
override_config_path="$experiment_dir/results/nemo-toHF-config.yaml"

mkdir -p "$nemo_output_dir" "$hf_output_dir"

# Use sed to replace 'cfg' with 'model' at the start of the block
sed 's/^cfg:/model:/' "$hparams_file" > "$override_config_path"

# Loop through all checkpoint files
for filepath in $(ls "$checkpoint_dir" | sort -t'=' -k2 -n); do
    # Extract the file name
    filename=$(basename "$filepath")

    # Extract steps and consumed_samples from the filename
    steps=$(echo "$filename" | sed -n 's/.*step=\([0-9]*\)-.*/\1/p')
    consumed=$(echo "$filename" | sed -n 's/.*consumed_samples=\([0-9]*\).0/\1/p')

    # For testing
    # steps=5625
    # consumed=675000
    # filename="megatron_llama_3_1_1.5B-step=$steps-consumed_samples=$consumed.0"

    # Form the nemo output path and Hugging Face output path
    nemo_file_path="$nemo_output_dir/step=$steps-consumed=$consumed.nemo"
    hf_file_path="$hf_output_dir/step=$steps-consumed=$consumed.bin"

    # Check if the Nemo or Hugging Face file already exists, if yes, skip
    if [[ -f "$nemo_file_path" && -f "$hf_file_path" ]]; then
        echo "Both Nemo and Hugging Face checkpoints already exist for step=$steps, consumed_samples=$consumed. Skipping..."
        continue
    fi

    # Run the Megatron to NeMo conversion if Nemo checkpoint doesn't exist
    if [[ ! -f "$nemo_file_path" ]]; then
        echo "Converting Megatron ckpt to Nemo ckpt: $filename"
        # export CUDA_VISIBLE_DEVICES=0
        python3 -m torch.distributed.launch --nproc_per_node=1 /users/xyixuan/NeMo/examples/nlp/language_modeling/megatron_ckpt_to_nemo.py \
            --checkpoint_folder "$checkpoint_dir" \
            --checkpoint_name "$filename" \
            --nemo_file_path "$nemo_file_path" \
            --model_type gpt \
            --hparams_file "$hparams_file" \
            --tensor_model_parallel_size 1 \
            --pipeline_model_parallel_size 1 \
            --gpus_per_node 1
    else
        echo "Nemo checkpoint already exists for step=$steps, consumed_samples=$consumed. Skipping NeMo conversion..."
    fi

    # Run the NeMo to Hugging Face conversion if HF checkpoint doesn't exist
    if [[ ! -f "$hf_file_path" ]]; then
        echo "Converting NeMo ckpt to Hugging Face ckpt: $nemo_file_path"
        python3 -u /users/xyixuan/NeMo/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py \
            --input_name_or_path="$nemo_file_path" \
            --input_tokenizer="$tokenizer_dir" \
            --output_path="$hf_file_path" \
            --override_config_path="$override_config_path"
    else
        echo "Hugging Face checkpoint already exists for step=$steps, consumed_samples=$consumed. Skipping HF conversion..."
    fi

    echo "Finished processing step=$steps, consumed_samples=$consumed"
done
