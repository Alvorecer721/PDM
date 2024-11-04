from numba import jit, prange
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
from dataclasses import replace

@dataclass
class Match:
    """Represents a matching substring between two sequences."""
    idx: int
    length: int
    start_pos1: int
    end_pos1: int
    start_pos2: int
    end_pos2: int
    values: np.ndarray
    
    def __repr__(self):
        return (
            f"Match(idx={self.idx}, "
            f"length={self.length}, "
            f"s1[{self.start_pos1}:{self.end_pos1}], "
            f"s2[{self.start_pos2}:{self.end_pos2}]), "
            f"values={self.values})"
        )

@jit(nopython=True)
def _compute_dp_matrix_2d(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """
    Compute the dynamic programming matrix for substring matching.
    
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
                dp[i, j] = dp[i-1, j-1] + 1
                
    return dp


@jit(nopython=True, parallel=True)
def _compute_dp_matrices_3d(s1_batch: np.ndarray, s2_batch: np.ndarray) -> np.ndarray:
    """
    Compute DP matrices for all pairs in the batch in parallel.
    """
    n_samples = len(s1_batch)
    m, n = s1_batch.shape[1], s2_batch.shape[1]
    dp_matrices = np.zeros((n_samples, m + 1, n + 1), dtype=np.int32)
    
    # Remove f-strings from Numba-compiled code
    for b in prange(n_samples):
        dp_matrices[b] = _compute_dp_matrix_2d(s1_batch[b], s2_batch[b])

    return dp_matrices


@jit(nopython=True)
def _find_potential_matches(dp: np.ndarray, min_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all potential matches from the DP matrix.
    Returns arrays of lengths and positions for matches.
    """
    m, n = dp.shape
    # Pre-allocate maximum possible size
    max_matches = (m-1) * (n-1)
    lengths = np.zeros(max_matches, dtype=np.int32)
    starts1 = np.zeros(max_matches, dtype=np.int32)
    starts2 = np.zeros(max_matches, dtype=np.int32)
    ends1 = np.zeros(max_matches, dtype=np.int32)
    
    match_idx = 0
    for i in range(1, m):
        for j in range(1, n):
            length = dp[i, j]
            if length >= min_length:
                lengths[match_idx] = length
                starts1[match_idx] = i - length
                ends1[match_idx] = i
                starts2[match_idx] = j - length
                match_idx += 1
    
    # Trim arrays to actual size
    return (lengths[:match_idx], starts1[:match_idx], 
            ends1[:match_idx], starts2[:match_idx])


@jit(nopython=True)
def _filter_overlapping_matches(lengths: np.ndarray, starts1: np.ndarray, 
                              ends1: np.ndarray, starts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter overlapping matches, prioritizing only by length.
    """
    n_matches = len(lengths)
    if n_matches == 0:
        return lengths, starts1, ends1, starts2
    
    # Sort indices by length only (descending)
    indices = np.argsort(-lengths)
    
    valid_mask = np.ones(n_matches, dtype=np.bool_)
    m = np.max(ends1) + 1
    n = np.max(starts2 + lengths) + 1
    used_positions = np.zeros((m, n), dtype=np.bool_)
    
    for i in range(n_matches):
        idx = indices[i]
        if not valid_mask[idx]:
            continue
            
        s1, e1 = starts1[idx], ends1[idx]
        s2 = starts2[idx]
        length = lengths[idx]
        
        overlap = False
        for x in range(s1, e1):
            for y in range(s2, s2 + length):
                if used_positions[x, y]:
                    overlap = True
                    break
            if overlap:
                break
                
        if not overlap:
            for x in range(s1, e1):
                for y in range(s2, s2 + length):
                    used_positions[x, y] = True
        else:
            valid_mask[idx] = False
    
    # Get the filtered arrays in length-sorted order
    valid_indices = indices[valid_mask[indices]]
    return (lengths[valid_indices], starts1[valid_indices],
            ends1[valid_indices], starts2[valid_indices])


@jit(nopython=True)
def _find_longest_match(dp: np.ndarray) -> Tuple[int, int, int, int, int]:
    """
    Find the single longest match from the DP matrix directly.
    Returns (length, start_pos1, end_pos1, start_pos2, end_pos2)
    """
    m, n = dp.shape
    max_length = 0
    start_pos1 = end_pos1 = start_pos2 = end_pos2 = 0
    
    for i in range(1, m):
        for j in range(1, n):
            length = dp[i, j]
            if length > max_length:
                max_length = length
                end_pos1 = i
                end_pos2 = j
                start_pos1 = i - length
                start_pos2 = j - length
    
    return max_length, start_pos1, end_pos1, start_pos2, end_pos2


class CommonSubstringMatcher:
    def __init__(self, s1_batch, s2_batch):
        self.s1_batch = np.array(s1_batch)
        self.s2_batch = np.array(s2_batch)
        self.num_seq = len(s1_batch)
        self.dp_matrices = None  # Initialize as None
        self._compute_dp()  # Compute DP matrices during initialization
        self._matches: Optional[List[List[Match]]] = None
        
    def _compute_dp(self):
        """Compute and cache the Dynamic Programming matrices."""
        if self.dp_matrices is None:
            start_time = time.time()
            self.dp_matrices = _compute_dp_matrices_3d(self.s1_batch, self.s2_batch)
            print(f"Computed DP matrices in {time.time() - start_time:.2f} seconds.")
    
    def _find_all_matches(self, seq_idx: int, min_length: int = 2) -> List[Match]:
        """
        Find all longest common substrings for a specific sequence in the batch.
        Now uses Numba-accelerated functions for better performance.
        """
        s1 = self.s1_batch[seq_idx]
        s2 = self.s2_batch[seq_idx]
        dp = self.dp_matrices[seq_idx]
        
        # Find potential matches using Numba
        lengths, starts1, ends1, starts2 = _find_potential_matches(dp, min_length)
        
        # Filter overlapping matches using Numba
        lengths, starts1, ends1, starts2 = _filter_overlapping_matches(
            lengths, starts1, ends1, starts2)
        
        # Convert to Match objects
        matches = []
        for i in range(len(lengths)):
            match = Match(
                idx=seq_idx,
                length=lengths[i],
                start_pos1=starts1[i],
                end_pos1=ends1[i],
                start_pos2=starts2[i],
                end_pos2=starts2[i] + lengths[i],
                values=s1[starts1[i]:ends1[i]]
            )
            matches.append(match)
        
        return matches
    
    def get_all_matches(self, min_length: int = 2) -> List[List[Match]]:
        """
        Get all matches for all sequences in the batch, sorted by length in descending order.
        """
        all_matches = []
        for seq_idx in tqdm(range(self.num_seq), desc="Finding matches", unit="sequence"):
            batch_matches = self._find_all_matches(seq_idx, min_length)
            all_matches.append(batch_matches)
        
        self._matches = all_matches
        return all_matches
    
    def get_longest_matches(self) -> List[Optional[Match]]:
        """Get longest match for each sequence in the batch."""
        longest_matches = []
        
        for seq_idx in tqdm(range(self.num_seq), desc="Finding matches", unit="sequence"):
            dp = self.dp_matrices[seq_idx]
            length, start1, end1, start2, end2 = _find_longest_match(dp)
            
            if length >= 2:  # Only create match if length meets minimum threshold
                match = Match(
                    idx=seq_idx,
                    length=length,
                    start_pos1=start1,
                    end_pos1=end1,
                    start_pos2=start2,
                    end_pos2=end2,
                    values=None  # Skip values as requested
                )
                longest_matches.append(match)
            else:
                longest_matches.append(None)
                
        return longest_matches
    
    def get_match_sequences(self, min_length: int = 2) -> List[List[np.ndarray]]:
        """Get matching sequences for all matches in the batch."""
        all_matches = self.get_all_matches(min_length)
        return [[match.values for match in seq_matches] 
                for seq_matches in all_matches]