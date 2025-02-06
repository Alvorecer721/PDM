import pytest
import numpy as np
from datasets import load_dataset
import os
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
import torch
import glob
from pathlib import Path
import inspect


from src.infer.distributed_inference import load_model
from src.infer.utils import set_seed
from src.verbatim_eval.utils import load_inference_data
from src.gutenberg.create_excerpt import create_tokenize_fn
from src.vis.output_html import log_model_generations


def get_ckpt_path(expr_path: str | Path) -> tuple[Path, str]:
    """Get the checkpoint path with highest step number and experiment name.
    
    Args:
        expr_path: Path to the experiment directory
        
    Returns:
        tuple: (model_path, experiment_name) where
            - model_path is the Path object to the checkpoint with highest step
            - experiment_name is the name of the experiment directory
    """
    # Convert to Path object if string
    expr_path = Path(expr_path)
    
    # Get the NeMo2HF directory path
    nemo_dir = expr_path / 'results' / 'NeMo2HF'
    
    # Get all checkpoint files and find the one with highest step
    ckpt_files = list(nemo_dir.glob('step=*-consumed=*.bin'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {nemo_dir}")
        
    model_path = max(ckpt_files, key=lambda x: int(str(x).split('step=')[1].split('-')[0]))
    
    # Get experiment name
    expr_name = expr_path.name
    
    return model_path, expr_name


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
    global global_expr_name, global_expr_dir
    set_seed(42)

    config = AutoConfig.from_pretrained('/capstor/users/cscs/xyixuan/PDM/config/llama3_1.5B_config.json')
    
    # global_expr_dir = '/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_1984000'
    global_expr_dir = '/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_11971350'

    # model_path = '/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_10200350/results/NeMo2HF/step=170004-consumed=10200240.bin'
    # model_path = "/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_3968000/results/NeMo2HF/step=66132-consumed=3967920.bin"
    # model_path = '/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_3968000/results/NeMo2HF/step=66133-consumed=3967980.bin'

    model_path, global_expr_name = get_ckpt_path(global_expr_dir)

    model = load_model(
        config=config,
        model_path=model_path
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.model_max_length = 200_000
    
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
    
    return model, tokenizer, generator

# repetitions = np.array([1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128])
repetitions = np.array([128, 256, 512, 1024, 2048])

@pytest.fixture(params=[
    (rep, off) 
    for rep in repetitions
    for off in [0, 5, 10, 30, 50, 100]
], ids=lambda x: f"rep_{x[0]}_off_{x[1]}")
def model_setup(request, base_model_setup):
    """Setup data for each rep_count using the base model setup"""
    rep_count, offset = request.param
    model, tokenizer, generator = base_model_setup
    
    inference = load_inference_data(
        base_dir=f'{global_expr_dir}/inference',
        rep=rep_count,
        policy='greedy',
        offset=offset,
        len_prefix=500,
        len_suffix=500
    )
    
    return model, tokenizer, generator, inference, rep_count, offset

@pytest.mark.parametrize(
    "seq_idx", 
    list(range(0,500,100)),
    ids=lambda x: f"seq_{x}"
)
def test_tokenization_consistency(model_setup, seq_idx):
    """Test if tokenization is consistent between direct and pipeline approaches"""
    model, tokenizer, _, inference, rep_count, _ = model_setup
    tokenize_fn = create_tokenize_fn(tokenizer)
    
    prefix = tokenizer.decode(inference[seq_idx]['prefix'])
    new_prefix = tokenize_fn(prefix).input_ids
    old_preifx = inference[seq_idx]['prefix']

    assert (len(new_prefix) == len(old_preifx)), \
        f"Tokenization length is not consistent for repetition {rep_count}, sample {seq_idx}"
    
    if not np.array_equal(new_prefix, old_preifx):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(new_prefix, old_preifx, error_msg, tokenizer)


@pytest.mark.parametrize(
    "seq_idx", 
    list(range(0,500,100)),
    ids=lambda x: f"seq_{x}"
)
def test_prefix_suffix_consistency(model_setup, seq_idx):
    _, tokenizer, _, inference, rep_count, offset = model_setup 
    
    prefix = inference[seq_idx]['prefix']
    suffix = inference[seq_idx]['true_suffix']
    prefix_suffix = prefix + suffix

    # Load the Gutenberg dataset for the current repetition count
    gutenberg = load_dataset("json", data_files=f'/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg/rep_{rep_count}_token.jsonl', split='train')
    
    # Get the corresponding text from Gutenberg dataset
    gutenberg_tokens = gutenberg[seq_idx]['input_ids']
    
    # Extract the relevant portion from Gutenberg text using the offset
    gutenberg_slice = gutenberg_tokens[offset:offset + len(prefix_suffix)]
    
    # Compare the concatenated prefix+suffix with the Gutenberg text slice
    assert len(prefix_suffix) == len(gutenberg_slice), \
        f"Length mismatch for repetition {rep_count}, sample {seq_idx}: " \
        f"prefix_suffix={len(prefix_suffix)}, gutenberg={len(gutenberg_slice)}"
    
    # If lengths match but contents don't, provide detailed mismatch information
    if not np.array_equal(prefix_suffix, gutenberg_slice):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(prefix_suffix, gutenberg_slice, error_msg, tokenizer)
    

@pytest.mark.parametrize(
    "seq_idx", 
    list(range(0,500,100)),
    ids=lambda x: f"seq_{x}"
)
def test_pipeline_generation(model_setup, seq_idx):
    """
    Test correctness of model generation: 
    - Generate text using the huggingface pipeline
    - Compare it to the true suffix
    """
    _, tokenizer, generator, data, rep_count, offset = model_setup

    # Set seed before generation
    set_seed(42)
    
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
    generated_suffix_tokens = data[seq_idx]['generated_suffix']

    # # Log the generations with colored differences
    # log_model_generations(
    #     torch.tensor(pipeline_suffix_tokens),
    #     generated_suffix_tokens,
    #     tokenizer,
    #     rep_count,
    #     seq_idx,
    #     offset,
    #     output_dir='/capstor/users/cscs/xyixuan/PDM/results',
    #     expr_name=global_expr_name,
    # )
    
    if not np.array_equal(pipeline_suffix_tokens, generated_suffix_tokens):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(pipeline_suffix_tokens, generated_suffix_tokens, error_msg, tokenizer)


@pytest.mark.parametrize(
    "seq_idx", 
    list(range(0,500,100)),
    ids=lambda x: f"seq_{x}"
)
def test_pipeline_generation_with_true_prefix(model_setup, seq_idx):
    """
    Test correctness of model generation: 
    - Generate text using the huggingface pipeline
    - Compare it to the true suffix
    """
    _, tokenizer, generator, data, rep_count, offset = model_setup

    # Set seed before generation
    set_seed(42)
    
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
    true_suffix_tokens = data[seq_idx]['true_suffix']

    # Log the generations with colored differences
    log_model_generations(
        torch.tensor(pipeline_suffix_tokens),
        true_suffix_tokens,
        tokenizer,
        rep_count,
        seq_idx,
        offset,
        output_dir='/capstor/users/cscs/xyixuan/PDM/results',
        expr_name=global_expr_name,
    )
    
    if not np.array_equal(pipeline_suffix_tokens, true_suffix_tokens):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(pipeline_suffix_tokens, true_suffix_tokens, error_msg, tokenizer)

@pytest.mark.parametrize(
    "seq_idx", 
    list(range(0,500,100)),
    ids=lambda x: f"seq_{x}"
)
def test_model_generation(model_setup, seq_idx):
    """
    Test correctness of model generation via comparing it to huggingface text generation pipeline.
    """
    model, tokenizer, _, data, rep_count, offset = model_setup

    # Clear cache before each test
    torch.cuda.empty_cache()

    # Set seed before generation
    set_seed(42)

    prefix = torch.tensor([128000]+data[seq_idx]['prefix'], dtype=torch.long).unsqueeze(0).cuda()
    assert prefix.shape[1] == len(data[seq_idx]['prefix'])+1, f"Prefix length is {prefix.shape[1]}"  


    # Generate text using the huggingface pipeline
    with torch.no_grad():
        model_tokens = model.generate(
            prefix, 
            do_sample=False, 
            max_new_tokens=500,
            num_beams=1,
        )[0]

    model_suffix_tokens = model_tokens[-500:]

    # Retrieve the model-generated tokens    
    model_gen_suffix_tokens = data[seq_idx]['generated_suffix']
    
    if not torch.equal(model_suffix_tokens.cpu(), torch.tensor(model_gen_suffix_tokens)):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}, offset {offset}:\n"
        find_mismatch(model_suffix_tokens, model_gen_suffix_tokens, error_msg, tokenizer)


@pytest.mark.parametrize("seq_idx", [0], ids=lambda x: f"seq_{x}")
def test_model_generation_with_logging(model_setup, seq_idx):
    """
    Test model generation and log results with colored differences.
    """
    model, tokenizer, _, data, rep_count, offset = model_setup

    # Set seed before generation
    set_seed(42)

    salt = []
    # salt = [79689, 4477, 25] # Continuous writing:
    prefix = torch.tensor([128000]+salt+data[seq_idx]['prefix'], dtype=torch.long).unsqueeze(0).cuda()
    # prefix = torch.tensor(data[seq_idx]['prefix'], dtype=torch.long).unsqueeze(0).cuda()

    suffix_length = 500

    # Generate text using the model
    model_tokens = model.generate(
        prefix, 
        do_sample=False, 
        max_new_tokens=suffix_length,
        num_beams=1,
    )[0]

    model_suffix_tokens = model_tokens[-1*suffix_length:]
    true_suffix_tokens = data[seq_idx]['true_suffix']
    
    # Log the generations with colored differences
    # log_model_generations(
    #     model_suffix_tokens,
    #     true_suffix_tokens,
    #     tokenizer,
    #     rep_count,
    #     seq_idx,
    #     offset,
    #     output_dir='/capstor/users/cscs/xyixuan/PDM/results',
    #     expr_name=global_expr_name,
    # )
    
    # Still perform the original assertion
    if not torch.equal(model_suffix_tokens.cpu(), torch.tensor(true_suffix_tokens)):
        error_msg = f"\nMismatch found in repetition {rep_count}, sample {seq_idx}:\n"
        find_mismatch(model_suffix_tokens, true_suffix_tokens, error_msg, tokenizer)