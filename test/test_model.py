import pytest
import numpy as np
from datasets import load_dataset
import os
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
from src.infer.distributed_inference import load_model
from src.verbatim_eval.utils import load_inference_data
from src.gutenberg.create_excerpt import create_tokenize_fn
from src.vis.output_html import log_model_generations
import torch


def get_experiment_name(model_path: str) -> str:
    """Extract experiment name from model path using basename.
    
    Args:
        model_path: Full path to the model
        
    Returns:
        Experiment name from the last directory in path
    """
    # Get the directory containing the model file
    model_dir = os.path.dirname(model_path)
    # Get the experiment name which is the basename of the directory path
    expr_name = os.path.basename(model_dir)
    
    return expr_name if expr_name else "unknown_experiment"

def find_mismatch(seq1, seq2, error_msg, tokenizer):
    """
    Compare two sequences and provide detailed error message if they don't match.

    Args:
        new_prefix: First sequence to compare
        old_prefix: Second sequence to compare 
        rep_count: Repetition count for error reporting
        seq_idx: Sample index for error reporting
        tokenizer: Tokenizer for decoding token IDs
        
    Raises:
        AssertionError: If sequences don't match, with detailed mismatch info
    """
    error_msg += "-" * 80 + "\n"

    s = SequenceMatcher(None, seq1, seq2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            error_msg += f"Position {i1}:{i2} vs {j1}:{j2}\n"
            error_msg += f"Token IDs: {seq1[i1:i2]} -> {seq2[j1:j2]}\n"
            error_msg += f"Decoded text: '{tokenizer.decode(seq1[i1:i2])}' -> '{tokenizer.decode(seq2[j1:j2])}'\n"
            error_msg += "-" * 80 + "\n"

    raise AssertionError(error_msg)


# Create a session-scoped fixture for the model and tokenizer
@pytest.fixture(scope="session")
def base_model_setup():
    """Setup model, tokenizer and generator once for the entire test session"""
    global global_model_path
    
    config = AutoConfig.from_pretrained('/capstor/users/cscs/xyixuan/PDM/config/llama3_1.5B_config.json')
    
    global_model_path = '/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_11971350/results/NeMo2HF/step=199500-consumed=11970000.bin'
    # model_path = "/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_3968000/results/NeMo2HF/step=66132-consumed=3967920.bin"
    # model_path = '/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_3968000/results/NeMo2HF/step=66133-consumed=3967980.bin'

    model = load_model(
        config=config,
        model_path=global_model_path
    )
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.model_max_length = 200_000
    
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
    
    return model, tokenizer, generator

@pytest.fixture(params=[128, 256, 512, 1024, 2048])
def model_setup(request, base_model_setup):
    """Setup data for each rep_count using the base model setup"""
    rep_count = request.param
    model, tokenizer, generator = base_model_setup
    
    data = load_inference_data(
        base_dir='/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_3968000/inference',
        rep=rep_count,
        policy='greedy'
    )
    
    return model, tokenizer, generator, data, rep_count

@pytest.mark.parametrize("seq_idx", [0, 5, 10, 11, 20])
def test_tokenization_consistency(model_setup, seq_idx):
    """Test if tokenization is consistent between direct and pipeline approaches"""
    model, tokenizer, _, data, rep_count = model_setup
    tokenize_fn = create_tokenize_fn(tokenizer)
    
    prefix = tokenizer.decode(data[seq_idx]['prefix'])
    new_prefix = tokenize_fn(prefix).input_ids
    old_preifx = data[seq_idx]['prefix']

    assert (len(new_prefix) == len(old_preifx)), \
        f"Tokenization length is not consistent for repetition {rep_count}, sample {seq_idx}"
    
    if not np.array_equal(new_prefix, old_preifx):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(new_prefix, old_preifx, error_msg, tokenizer)
    

@pytest.mark.parametrize("seq_idx", [0, 5, 10, 11, 20])
def test_pipeline_generation(model_setup, seq_idx):
    """
    Test correctness of model generation: 
    - Generate text using the huggingface pipeline
    - Compare it to the true suffix
    """
    model, tokenizer, generator, data, rep_count = model_setup
    
    prefix = tokenizer.decode(data[seq_idx]['prefix'])

    # Generate text using the huggingface pipeline
    pipeline_tokens = generator(
        prefix, 
        do_sample=False, 
        max_new_tokens=500,
        num_beams=1,
        return_tensors=True
    )[0]['generated_token_ids']

    pipeline_suffix_tokens = pipeline_tokens[-500:]

    # Retrieve the model-generated tokens    
    ref_tokens = data[seq_idx]['true_suffix']
    
    if not np.array_equal(pipeline_suffix_tokens, ref_tokens):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(pipeline_suffix_tokens, ref_tokens, error_msg, tokenizer)


@pytest.mark.parametrize("seq_idx", [0, 5, 10, 11, 20])
def test_model_generation(model_setup, seq_idx):
    """
    Test correctness of model generation via comparing it to huggingface text generation pipeline.
    """
    model, tokenizer, _, data, rep_count = model_setup

    prefix = torch.tensor([128000]+data[seq_idx]['prefix'], dtype=torch.long).unsqueeze(0).cuda()

    # Generate text using the huggingface pipeline
    model_tokens = model.generate(
        prefix, 
        do_sample=False, 
        max_new_tokens=500,
        num_beams=1,
    )[0]

    model_suffix_tokens = model_tokens[-500:]

    # Retrieve the model-generated tokens    
    model_gen_suffix_tokens = data[seq_idx]['generated_suffix']
    
    if not np.array_equal(model_suffix_tokens, model_gen_suffix_tokens):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(model_suffix_tokens, model_gen_suffix_tokens, error_msg, tokenizer)


@pytest.mark.parametrize("seq_idx", [23, 40, 102, 153, 254, 277])
def test_model_generation_with_logging(model_setup, seq_idx):
    """
    Test model generation and log results with colored differences.
    """
    model, tokenizer, _, data, rep_count = model_setup

    # Extract experiment name from model path
    expr_name = get_experiment_name(global_model_path)

    prefix = torch.tensor([128000]+data[seq_idx]['prefix'], dtype=torch.long).unsqueeze(0).cuda()
    # prefix = torch.tensor(data[seq_idx]['prefix'], dtype=torch.long).unsqueeze(0).cuda()

    # Generate text using the model
    model_tokens = model.generate(
        prefix, 
        do_sample=False, 
        max_new_tokens=500,
        num_beams=1,
    )[0]

    model_suffix_tokens = model_tokens[-500:]
    true_suffix_tokens = data[seq_idx]['true_suffix']
    
    # Log the generations with colored differences
    log_model_generations(
        model_suffix_tokens,
        true_suffix_tokens,
        tokenizer,
        rep_count,
        seq_idx,
        output_dir='/capstor/users/cscs/xyixuan/PDM/results',
        expr_name=expr_name,
    )
    
    # Still perform the original assertion
    if not np.array_equal(model_suffix_tokens, true_suffix_tokens):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(model_suffix_tokens, true_suffix_tokens, error_msg, tokenizer)