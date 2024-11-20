import numpy as np
from numba import njit
from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class RelaxedMatch:
    length: int
    edits: int
    start_pos1: int
    end_pos1: int
    start_pos2: int
    end_pos2: int
    values1: np.ndarray
    values2: np.ndarray

    def __repr__(self):
        return (
            f"RelaxedMatch(length={self.length}, "
            f"edits={self.edits}, "
            f"s1[{self.start_pos1}:{self.end_pos1}]={self.values1}, "
            f"s2[{self.start_pos2}:{self.end_pos2}]={self.values2})"
        )


@njit
def _dp_edit_distance(s1: np.ndarray, s2: np.ndarray, max_edits: int) -> np.ndarray:
    """Compute edit distance with dynamic programming."""
    m, n = len(s1), len(s2)
    dp = np.full((m + 1, n + 1), max_edits + 1, dtype=np.int32)
    
    # Initialize base cases
    dp[0, 0] = 0
    for i in range(1, m + 1):
        dp[i, 0] = i
    for j in range(1, n + 1):
        dp[0, j] = j
        
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                # Regular operations (insertion, deletion, substitution)
                dp[i, j] = min(
                    dp[i-1, j] + 1,    # deletion
                    dp[i, j-1] + 1,    # insertion
                    dp[i-1, j-1] + 1   # substitution
                )
                # Check for transposition
                if (i > 1 and j > 1 and 
                    s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]):
                    dp[i, j] = min(dp[i, j], dp[i-2, j-2] + 1)
    
    return dp


@njit
def _find_match_with_edits(s1: np.ndarray, s2: np.ndarray, max_edits: int) -> Tuple[int, int]:
    """Find the longest match with up to max_edits edits."""
    dp = _dp_edit_distance(s1, s2, max_edits)
    m, n = len(s1), len(s2)
    
    # Get edit distance
    edits = dp[m, n]
    if edits > max_edits:
        return 0, max_edits + 1
        
    return max(m, n), edits


def find_all_potential_matches(
    s1: np.ndarray,
    s2: np.ndarray,
    min_length: int = 3,
    max_edits: int = 1
) -> List[RelaxedMatch]:
    """Find all potential relaxed matches between s1 and s2.
    
    Args:
        s1: First sequence
        s2: Second sequence
        min_length: Minimum length of a match
        max_edits: Maximum number of edits allowed
        
    Returns:
        List of potential matches, sorted by length (descending) and edits (ascending)
    """
    matches = []
    m, n = len(s1), len(s2)
    
    # Try different window sizes
    for window in range(min_length, max(m, n) + 1):
        # Try all possible starting positions
        for i in range(m - min_length + 1):
            for j in range(n - min_length + 1):
                # Get subsequences to compare
                end1 = min(i + window, m)
                end2 = min(j + window, n)
                seq1 = s1[i:end1]
                seq2 = s2[j:end2]
                
                # Skip if length difference is too large
                if abs(len(seq1) - len(seq2)) > max_edits:
                    continue
                
                match_length, num_edits = _find_match_with_edits(seq1, seq2, max_edits)
                
                if match_length >= min_length and num_edits <= max_edits:
                    matches.append(RelaxedMatch(
                        length=match_length,
                        edits=num_edits,
                        start_pos1=i,
                        end_pos1=end1,
                        start_pos2=j,
                        end_pos2=end2,
                        values1=s1[i:end1],
                        values2=s2[j:end2]
                    ))
    
    # Sort matches by length (descending) and edits (ascending)
    matches.sort(key=lambda m: (-m.length, m.edits))
    return matches

def filter_contained_matches(matches: List[RelaxedMatch]) -> List[RelaxedMatch]:
    """Filter out matches that are contained within longer matches with fewer or equal edits.
    
    Args:
        matches: List of matches, should be sorted by length (descending) and edits (ascending)
        
    Returns:
        Filtered list of matches with contained matches removed
    """
    final_matches = []
    for match in matches:
        # Check if this match is contained in any longer match
        is_contained = False
        for longer_match in final_matches:
            if (match.start_pos1 >= longer_match.start_pos1 and 
                match.end_pos1 <= longer_match.end_pos1 and
                match.edits >= longer_match.edits):
                is_contained = True
                break
        
        if not is_contained:
            final_matches.append(match)
    
    return final_matches

def find_relaxed_matches(
    s1: np.ndarray,
    s2: np.ndarray,
    min_length: int = 3,
    max_edits: int = 1
) -> List[RelaxedMatch]:
    """Find all relaxed matches between s1 and s2.
    
    This function combines find_all_potential_matches() and filter_contained_matches()
    to find all valid matches and remove those contained in longer matches.
    
    Args:
        s1: First sequence
        s2: Second sequence
        min_length: Minimum length of a match
        max_edits: Maximum number of edits allowed
        
    Returns:
        List of filtered matches, sorted by length (descending) and edits (ascending)
    """
    potential_matches = find_all_potential_matches(s1, s2, min_length, max_edits)
    return filter_contained_matches(potential_matches)


if __name__ == "__main__":
    from datasets import load_dataset
    from time import time

    # Load the JSONL file
    # dataset = load_dataset('json', 
    #                     data_files='/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Goldfish_K_5_H_13_GBS_120_EPOCH_93/step=3000-consumed=360000/rank0.jsonl',
    #                     split='train')

    # s1 = dataset[0]['true_suffix']
    # s2 = dataset[0]['generated_suffix']

    
    s1 = np.array([1, 2, 4, 3, 5, 6, 7, 11, 9, 10, 12, 13, 15, 14, 16, 17, 18, 20, 19])
    s2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20])

    start = time()
    result = find_relaxed_matches(s1, s2, min_length=10, max_edits=2)
    print(f"Found {len(result)} matches in {time() - start:.2f}s")

    for res in result:
        print(res)