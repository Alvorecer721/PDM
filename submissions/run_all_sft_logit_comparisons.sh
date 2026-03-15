#!/bin/bash
# Submit logit comparison jobs for all SFT checkpoints
# Compares latest Megatron checkpoint vs HF checkpoint for each run

SLURM_SCRIPT="/iopsstor/scratch/cscs/rkreft/PDM/submissions/submit-compare-logits.slurm"
BASE="/users/rkreft/megatron-repo/logs/Meg-Runs/apertus_image_extension"
MEGATRON_DIR="/users/rkreft/megatron-repo"
TOKENIZER="/users/rkreft/MLLM-infra01-folder/tokenizer/apertus_emu3.5"

submit_comparison() {
    local name="$1"
    local meg_ckpt="$2"
    local hf_dir="$3"

    echo "Submitting: $name"
    echo "  MEG: $meg_ckpt"
    echo "  HF:  $hf_dir"

    sbatch --job-name="cmp-${name:0:40}" \
        "$SLURM_SCRIPT" \
        "hf:${hf_dir}" \
        "meg:${meg_ckpt}" \
        --megatron-dir "$MEGATRON_DIR" \
        --tokenizer "$TOKENIZER" \
        --old-megatron

    echo ""
}

# 1) apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss
submit_comparison \
    "SFT-32n-s2onlytxtloss" \
    "${BASE}/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF"

# 2) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192
submit_comparison \
    "SFT-64n-img-text" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192/HF"

# 3) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss-S2ImgWeight0.1
submit_comparison \
    "SFT-64n-s2txtloss-ImgW0.1" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss-S2ImgWeight0.1/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss-S2ImgWeight0.1/HF"

# 4) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss-S2noS1
submit_comparison \
    "SFT-64n-s2txtloss-S2noS1" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss-S2noS1/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss-S2noS1/HF"

# 5) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight0-s2BlipExt-imW0
submit_comparison \
    "SFT-64n-ImgW0-BlipExt" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight0-s2BlipExt-imW0/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight0-s2BlipExt-imW0/HF"

# 6) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight0.1-until8030-then-Imw0
submit_comparison \
    "SFT-64n-ImgW0.1-then-0" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight0.1-until8030-then-Imw0/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight0.1-until8030-then-Imw0/HF"

# 7) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-s2BlipExt-imW0
submit_comparison \
    "SFT-64n-ImgW1-BlipExt" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-s2BlipExt-imW0/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-s2BlipExt-imW0/HF"

# 8) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-until8030-then-decay
submit_comparison \
    "SFT-64n-ImgW1-then-decay" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-until8030-then-decay/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-until8030-then-decay/HF"

# 9) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-until8030-then-imw0
submit_comparison \
    "SFT-64n-ImgW1-then-imw0" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-until8030-then-imw0/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeight1-until8030-then-imw0/HF"

# 10) apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeightDecay-cosine-max0.1-min0.0-start0.0-end1.0
submit_comparison \
    "SFT-64n-ImgWDecay-cosine" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeightDecay-cosine-max0.1-min0.0-start0.0-end1.0/checkpoints" \
    "${BASE}/apertus-8b-img-SFT-64nodes-gbs512-mbs1-steps8030-seqlen8192-S2ImgWeightDecay-cosine-max0.1-min0.0-start0.0-end1.0/HF"

# 11) apertus-8b-img-SFT_TXTONLY-16nodes-gbs512-mbs1-steps1116-img-text-seqlen8192-s2onlytxtloss
submit_comparison \
    "SFT-TXTONLY-16n-s2txtloss" \
    "${BASE}/apertus-8b-img-SFT_TXTONLY-16nodes-gbs512-mbs1-steps1116-img-text-seqlen8192-s2onlytxtloss/checkpoints" \
    "${BASE}/apertus-8b-img-SFT_TXTONLY-16nodes-gbs512-mbs1-steps1116-img-text-seqlen8192-s2onlytxtloss/HF"

# 12) apertus-8b-img-SFT_TXTONLY-16nodes-gbs512-mbs1-steps1116-img-text-seqlen8192-s2onlytxtloss-S2noS1
submit_comparison \
    "SFT-TXTONLY-16n-S2noS1" \
    "${BASE}/apertus-8b-img-SFT_TXTONLY-16nodes-gbs512-mbs1-steps1116-img-text-seqlen8192-s2onlytxtloss-S2noS1/checkpoints" \
    "${BASE}/apertus-8b-img-SFT_TXTONLY-16nodes-gbs512-mbs1-steps1116-img-text-seqlen8192-s2onlytxtloss-S2noS1/HF"

echo "All 12 logit comparison jobs submitted!"

sbatch --job-name="cmp-newHF-vs-meg" "$SLURM_SCRIPT" \
    "hf:/users/rkreft/scratch/apertus1p5mainrun8bs1/HF" \
    "meg:/capstor/store/cscs/swissai/infra01/apertus_1p5/Megatron-LM/logs/Meg-Runs/main-runs-apertus-1p5/apertus-1p5-8b/checkpoints" \
    --megatron-dir /users/rkreft/scratch/Megatron2/Megatron-LM

sbatch --job-name="cmp-newHF-vs-meg" "$SLURM_SCRIPT" \
    "hf:/capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/HF_CKPT_96000_nemo" \
    "meg:/capstor/store/cscs/swissai/infra01/apertus_1p5/Megatron-LM/logs/Meg-Runs/main-runs-apertus-1p5/apertus-1p5-8b/checkpoints" \
    --megatron-dir /users/rkreft/scratch/Megatron2/Megatron-LM