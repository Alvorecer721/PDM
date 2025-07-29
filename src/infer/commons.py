import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import random
import logging
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from datasets import Features, Sequence, Value
import platform

import sys
# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verbatim_eval.LCS import find_longest_common_substrings
from verbatim_eval.my_rouge import compute_rouge_l_2d, _compute_dp_matrix_2d

# ------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def is_rank_0():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_distributed(seed):
    set_seed(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


# ------------------------------------------------------------------------------------
# Dataset processing
# ------------------------------------------------------------------------------------
def batch_processing_gutenberg(batch, _prefix_len, _offset, _suffix_len=None):
    if _suffix_len is None:
        _suffix_len = _prefix_len

    prefix_suffix_list = []
    for sequence in batch["input_ids"]:
        assert _offset + _prefix_len + _suffix_len <= len(sequence), (
            f"Requested offset ({_offset}), prefix length ({_prefix_len}), "
            f"and suffix length ({_suffix_len}) exceed sequence length ({len(sequence)})."
        )
        prefix_suffix = sequence[_offset:_offset + _prefix_len + _suffix_len]
        prefix_suffix_list.append(prefix_suffix)

    return {"prefix_suffix": prefix_suffix_list}


def process_dataset(
    data_path, batch_processing_fn, prefix_length, suffix_length, offset, num_proc
):
    logger.info(f"Processing dataset from {data_path}")

    # dataset = ydataset(
    #     "json",
    #     data_files=str(data_path),
    #     split="train",
    #     cache_dir="/iopsstor/scratch/cscs/xyixuan/cache"
    # )

    features = Features({"input_ids": Sequence(Value("int64"))})
    arch = platform.machine()

    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
        cache_dir=f"/iopsstor/scratch/cscs/xyixuan/cache/{arch}",
        features=features,
    )

    processed_dataset = dataset.map(
        batch_processing_fn,
        batched=True,
        desc="Generating prefix and suffix pairs",
        num_proc=num_proc,
        fn_kwargs={
            "_prefix_len": prefix_length,
            "_suffix_len": suffix_length,
            "_offset": offset,
        },
    )
    
    # Extract the prefix_suffix column
    processed_dataset = processed_dataset["prefix_suffix"]

    logger.info(f"Processed {len(processed_dataset)} samples")
    return processed_dataset


# ------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------
def setup_output_directories(experiment_path, offset, prefix_length, suffix_length):
    output_dir = experiment_path / "inference"
    output_path = output_dir / f"offset_{offset}_prefix_{prefix_length}_suffix_{suffix_length}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_inference_dir(output_path, rep, gen_policy):
    return output_path / f"rep_{rep}_{gen_policy}"


# ------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------
def calculate_text_metrics(true_seq, gen_seq):
    ttr_ref = len(set(true_seq)) / len(true_seq) if true_seq else 0
    ttr_gen = len(set(gen_seq)) / len(gen_seq) if gen_seq else 0

    dp_matrix = _compute_dp_matrix_2d(true_seq, gen_seq)
    rouge_l = compute_rouge_l_2d(dp_matrix)
    del dp_matrix

    return {
        "TTR_ref": ttr_ref,
        "TTR_gen": ttr_gen,
        "Rouge-L": rouge_l
    }


def process_sequences(sequences, batch_tensor, prefix_length, suffix_length, prepend_tokens):
    prefixes = batch_tensor[:, :prefix_length].cpu().tolist()
    true_suffixes = batch_tensor[:, prefix_length:].cpu().tolist()
    generated_suffixes = (
        sequences[:, prefix_length + len(prepend_tokens):].cpu().tolist()
    )  # Skip prepend BOS token

    lcs_result = find_longest_common_substrings(true_suffixes, generated_suffixes)
    lcs = lcs_result['max_length'].to_numpy()
    lcs_norm = lcs / suffix_length

    return prefixes, true_suffixes, generated_suffixes, lcs_norm


def write_results(jsonl_file, prefixes, true_suffixes, generated_suffixes,
                  seq_nlls_mean, seq_nlls_std, perplexity, lcs_norm,
                  ref_nll_mean, ref_nll_std, ref_perplexity):

    nll_means = seq_nlls_mean.cpu().tolist()
    nll_stds = seq_nlls_std.cpu().tolist()
    perplexities = perplexity.cpu().tolist()
    ref_nll_means = ref_nll_mean.cpu().tolist()
    ref_nll_stds = ref_nll_std.cpu().tolist()
    ref_perplexities = ref_perplexity.cpu().tolist()

    for p, t, g, nll_m, nll_s, lcs_n, ppl, ref_nll_m, ref_nll_s, ref_ppl in zip(
        prefixes, true_suffixes, generated_suffixes, nll_means, nll_stds,
        lcs_norm, perplexities, ref_nll_means, ref_nll_stds, ref_perplexities
    ):
        metrics = calculate_text_metrics(t, g)
        result = {
            "prefix": p,
            "true_suffix": t,
            "generated_suffix": g,
            "nll_mean": nll_m,
            "nll_std": nll_s,
            "perplexity": ppl,
            "ref_nll_mean": ref_nll_m,
            "ref_nll_std": ref_nll_s,
            "ref_perplexity": ref_ppl,
            "lcs_norm": lcs_n,
            **metrics
        }
        json.dump(result, jsonl_file)
        jsonl_file.write("\n")
        jsonl_file.flush()


# ------------------------------------------------------------------------------------
# NLL (reference + generated; forward-pass)
# ------------------------------------------------------------------------------------
@torch.no_grad()
def calc_reference_nll(model, input_tensor, suffix_length):
    inputs = input_tensor[:, :-1]
    targets = input_tensor[:, 1:]

    outputs = model(inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    criterion = nn.CrossEntropyLoss(reduction="none")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    token_nlls_flat = criterion(flat_logits, flat_targets)
    token_nlls = token_nlls_flat.reshape(targets.shape)

    suffix_token_nlls = token_nlls[:, -suffix_length:]
    seq_nlls_mean = suffix_token_nlls.mean(dim=1)
    seq_nlls_std = suffix_token_nlls.std(dim=1)
    ppl_ref = torch.exp(seq_nlls_mean)

    del flat_logits, flat_targets, token_nlls_flat, logits, outputs
    torch.cuda.empty_cache()

    return suffix_token_nlls, seq_nlls_mean, seq_nlls_std, ppl_ref


@torch.no_grad()
def calc_generation_nll_forward(model, sequences, suffix_length):
    """
    sequences: [B, bos + prefix + suffix]
    """
    model.eval()

    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]

    outputs = model(inputs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    criterion = nn.CrossEntropyLoss(reduction="none")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    token_nlls_flat = criterion(flat_logits, flat_targets)
    token_nlls_full = token_nlls_flat.reshape(targets.shape)  # [B, T-1]

    suffix_token_nlls = token_nlls_full[:, -suffix_length:]
    seq_nlls_mean = suffix_token_nlls.mean(dim=1)
    seq_nlls_std = suffix_token_nlls.std(dim=1)
    ppl = torch.exp(seq_nlls_mean)

    del flat_logits, flat_targets, token_nlls_flat, outputs, logits
    torch.cuda.empty_cache()

    return suffix_token_nlls, seq_nlls_mean, seq_nlls_std, ppl


# ------------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------------
def load_model(model_path: Path):
    logger.info(f"Loading model from {model_path}")
    if not model_path.exists():
        raise ValueError(f"Model checkpoint not found at {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )


def run(
    model,
    dataset,
    prefix_length,
    suffix_length,
    batch_size,
    inference_dir,
    policy,
    seed,
    num_beams,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    model.to(local_rank)
    model.eval()

    setup_distributed(seed)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda batch: batch
    )

    inference_dir.mkdir(parents=True, exist_ok=True)
    output_file = inference_dir / f"rank{rank}.jsonl"

    generation_configs = {
        "greedy": {
            "num_beams": 1,
            "do_sample": False,
            "num_return_sequences": 1,
        },
        "nucleus": {
            "num_beams": 1,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.9,
            "num_return_sequences": 1,
        },
        "beam": {
            "num_beams": num_beams,
            "do_sample": False,
            "num_return_sequences": 1,
        },
    }

    with open(output_file, "w") as jsonl_file:
        for batch in tqdm(
            dataloader,
            desc=f"Generating Suffix (Rank {rank}/{world_size-1})",
            unit="batch",
            ncols=100,
            disable=rank != 0,
        ):
            torch.cuda.empty_cache()

            batch_tensor = torch.tensor(batch, device=local_rank)

            # Prepend <BoS> token
            prepend_tokens = torch.tensor([model.config.bos_token_id], device=batch_tensor.device)
            input_with_bos = torch.cat(
                [
                    prepend_tokens.repeat(batch_tensor.shape[0], 1),
                    batch_tensor,
                ],
                dim=1,
            )

            # Reference NLL (gold suffix)
            _, ref_nll_mean, ref_nll_std, ref_perplexity = calc_reference_nll(
                model, input_with_bos, suffix_length
            )

            # BOS + prefix for generation
            input_with_bos = input_with_bos[:, : 1 + prefix_length]

            assert input_with_bos.shape[1] == prefix_length + len(prepend_tokens), \
                f"Input shape mismatch: {input_with_bos.shape}"
            assert batch_tensor.shape[1] == prefix_length + suffix_length, \
                f"Batch shape mismatch: {batch_tensor.shape}"

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_with_bos,
                    max_new_tokens=suffix_length,
                    min_new_tokens=suffix_length,
                    return_dict_in_generate=True,
                    output_scores=False,  # not needed anymore
                    **generation_configs[policy],
                )

            sequences = outputs.sequences  # [B, bos + prefix + suffix]

            # Compute NLL for generated suffix with a forward pass
            _, seq_nlls_mean, seq_nlls_std, perplexity = calc_generation_nll_forward(
                model, sequences, suffix_length
            )

            assert sequences.shape[1] == len(prepend_tokens) + prefix_length + suffix_length, \
                f"Output shape mismatch: {sequences.shape}"

            prefixes, true_suffixes, generated_suffixes, lcs_norm = process_sequences(
                sequences, batch_tensor, prefix_length, suffix_length, prepend_tokens
            )

            write_results(
                jsonl_file, prefixes, true_suffixes, generated_suffixes,
                seq_nlls_mean, seq_nlls_std, perplexity, lcs_norm,
                ref_nll_mean, ref_nll_std, ref_perplexity
            )

            # Cleanup
            del batch_tensor, sequences, outputs, input_with_bos
            del seq_nlls_mean, seq_nlls_std, perplexity
            del ref_nll_mean, ref_nll_std, ref_perplexity
            del prefixes, true_suffixes, generated_suffixes, lcs_norm
            torch.cuda.empty_cache()

    dist.barrier()
