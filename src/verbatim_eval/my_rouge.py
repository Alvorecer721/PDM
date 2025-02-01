from numba import jit
import numpy as np
import pandas as pd
import logging
from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
TOKENIZER.model_max_length = 200_000

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _compute_dp_matrix_2d(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """
    Compute the dynamic programming matrix for rouge-l calculation.
    
    Parameters
    ----------
    s1 : np.ndarray
        First input sequence
    s2 : np.ndarray
        Second input sequence
        
    Returns
    -------
    np.ndarray
        Dynamic programming matrix where dp[i,j] represents the length of
        the common substring ending at s1[i-1] and s2[j-1]
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp
                

@jit(nopython=True)
def _backtrack_lcs_2d(ref, pred, dp):
   m, n = len(ref), len(pred)
   contributing_tokens = np.zeros(dp[m][n], dtype=np.int32)
   contributing_pos1 = np.zeros(dp[m][n], dtype=np.int32) 
   contributing_pos2 = np.zeros(dp[m][n], dtype=np.int32)

   i, j = m, n
   k = dp[m][n] - 1
   
   while i > 0 and j > 0:
       if ref[i-1] == pred[j-1]:
           contributing_tokens[k] = ref[i-1]
           contributing_pos1[k] = i-1
           contributing_pos2[k] = j-1
           i -= 1
           j -= 1
           k -= 1
       elif dp[i-1][j] >= dp[i][j-1]:
           i -= 1
       else:
           j -= 1

   return dp[m][n], contributing_tokens, contributing_pos1, contributing_pos2


def compute_rouge_l_2d(dp):
   """Compute ROUGE-L score from dp matrix."""
   ref_len, pred_len = dp.shape[0]-1, dp.shape[1]-1
   lcs_length = dp[ref_len][pred_len]
   precision = lcs_length / pred_len if pred_len else 0 
   recall = lcs_length / ref_len if ref_len else 0
   f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
   return f1


def find_contributing_tokens(ref, pred):
    """Find tokens contributing to LCS."""
    dp = _compute_dp_matrix_2d(ref, pred)
    return _backtrack_lcs_2d(ref, pred, dp)


def show_contributing_tokens(tokens):
    """Display contributing tokens."""
    unique_tokens, counts = np.unique(tokens, return_counts=True)
    print(f"\nRouge L token counts: {len(tokens)}")
    print("-" * 50)
    print("Token ID  | Text            | Count")
    print("-" * 50)
    for token, count in zip(unique_tokens, counts):
        token_text = TOKENIZER.decode(token).replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        print(f"{token:>8} | {token_text:<15} | {count:>5}")


if __name__ == "__main__":
    from utils import load_inference_data, find_top_quantile_indices
    from src.verbatim_eval.rouge_ttr import eval_rouge_ttr, log_metric

    goldfish_res_greedy = eval_rouge_ttr(
        "/iopsstor/scratch/cscs/xyixuan/experiment", 
        experiments=[
            "llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_1984000",
            "llama_1.5B_Sparse_Gutenberg_Standard_GBS_60_SEQ_1984000"
        ], 
        repetitions=[0],
        len_suffix=500,
        policy='greedy'
    )

    # Get the scores from your dictionary
    TTR_scores = goldfish_res_greedy['llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_1984000'][0]['TTR_gen']['scores']
    RougeL_scores = goldfish_res_greedy['llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_1984000'][0]['Rouge-L']['scores']

    # Find the indices
    top_indices = find_top_quantile_indices(TTR_scores, RougeL_scores, q=0.2)

    assert len(top_indices) > 0, "No top indices found"

    print(f"Indice: {top_indices[0]}")

    temp = load_inference_data(
        base_dir="/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_1984000/inference",
        rep=0,
        policy='greedy'
    )

    ref = temp[top_indices[0]]['true_suffix']
    pred = temp[top_indices[0]]['generated_suffix']

    print(f"ROUGE-L: {compute_rouge_l_2d(_compute_dp_matrix_2d(ref, pred))}")

    _, tokens, _, _ = find_contributing_tokens(ref, pred)
    show_contributing_tokens(tokens)
