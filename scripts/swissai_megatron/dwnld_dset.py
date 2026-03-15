#!/usr/bin/env python3
"""
Script to download evaluation datasets for lm-evaluation-harness
"""
import os
import argparse
from datasets import load_dataset, get_dataset_config_names

# Track failures
failed_downloads = []

print("=" * 60)
print("Starting dataset downloads for lm-evaluation-harness")
print("=" * 60)
print("\nHuggingFace Cache Configuration:")
print(f"  HF_HOME:           {os.environ.get('HF_HOME', 'Not set (default: ~/.cache/huggingface)')}")
print(f"  HF_HUB_CACHE:      {os.environ.get('HF_HUB_CACHE', 'Not set (default: $HF_HOME/hub)')}")
print(f"  HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', 'Not set (default: $HF_HOME/datasets)')}")
print("=" * 60)

# ============================================================================
# Define all datasets
# ============================================================================

# MMLU - all 57 subsets
mmlu_configs = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]

# Standard Benchmarks
standard_benchmarks = [
    {
        "name": "HellaSwag",
        "dataset": "Rowan/hellaswag",
        "configs": [None]
    },
    {
        "name": "WinoGrande",
        "dataset": "winogrande",
        "configs": ["winogrande_xl"]
    },
    {
        "name": "WikiText",
        "dataset": "Salesforce/wikitext",
        "configs": ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]
    },
    {
        "name": "WikiText (Document Level)",
        "dataset": "EleutherAI/wikitext_document_level",
        "configs": ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]
    },
    {
        "name": "ARC",
        "dataset": "allenai/ai2_arc",
        "configs": ["ARC-Easy", "ARC-Challenge"]
    },
    {
        "name": "PIQA",
        "dataset": "baber/piqa",
        "configs": [None]
    },
    {
        "name": "CommonsenseQA",
        "dataset": "tau/commonsense_qa",
        "configs": [None]
    }
]

# Instruction Benchmarks (for --instruct mode)
instruction_benchmarks = [
    {
        "name": "GSM8K",
        "dataset": "gsm8k",
        "configs": ["main"]
    },
    {
        "name": "IFEval",
        "dataset": "google/IFEval",
        "configs": [None]
    },
    {
        "name": "TruthfulQA (Multiple Choice)",
        "dataset": "truthful_qa",
        "configs": ["multiple_choice"]
    },
    {
        "name": "BBH (Big-Bench Hard)",
        "dataset": "SaylorTwift/bbh",
        "configs": ["default"]
    }
]

multimodal_datasets = [

]




def parse_args():
    parser = argparse.ArgumentParser("Download HF datasets for benchmarking")
    parser.add_argument("--txt-pretrain", action="store_true", help="Download txt model pretrain benchmarks")
    parser.add_argument("--txt-instruct", action="store_true", help="Download text model instruct benchmarks")
    parser.add_argument("--mm", action="store_true", help="Download multi modal benchmarks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    datasets_to_dwnld = []
    dwnld_mmlu = False
    if args["txt-pretrain"]:
        datasets_to_dwnld += standard_benchmarks
        dwnld_mmlu = True
    if args["txt-instruct"]:
        datasets_to_dwnld += instruction_benchmarks
    if args["mm"]:
        datasets_to_dwnld += multimodal_datasets
 
    total_datasets = 1 if dwnld_mmlu else 0 + len(datasets_to_dwnld) 

    # Download MMLU
    if dwnld_mmlu:
        print(f"\n[1/{total_datasets}] Downloading MMLU (all subsets)...")
        for i, config in enumerate(mmlu_configs, 1):
            try:
                print(f"  [{i}/{len(mmlu_configs)}] {config}...", end=" ")
                load_dataset("cais/mmlu", config, trust_remote_code=True)
                print("✓")
            except Exception as e:
                print(f"✗ FAILED")
                failed_downloads.append(f"cais/mmlu ({config}): {str(e)[:100]}")

    # Download other datasets
    for idx, dataset_info in enumerate(datasets_to_dwnld, 2):
        name = dataset_info["name"]
        dataset = dataset_info["dataset"]
        configs = dataset_info["configs"]

        print(f"\n[{idx}/{total_datasets}] Downloading {name}...")
        
        for config in configs:
            config_str = config if config else "default"
            try:
                print(f"  - {config_str}...", end=" ")
                load_dataset(dataset, config, trust_remote_code=True)
                print("✓")
            except Exception as e:
                print(f"✗ FAILED")
                failed_downloads.append(f"{dataset} ({config_str}): {str(e)[:100]}")
                
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    if not failed_downloads:
        print("\n✓ All datasets downloaded successfully!")
    else:
        print(f"\n✗ {len(failed_downloads)} download(s) failed:\n")
        for i, failure in enumerate(failed_downloads, 1):
            print(f"{i}. {failure}")

    print("\n" + "=" * 60)
