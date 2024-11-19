import numpy as np
from numba import jit
from enum import Enum
from typing import List, Tuple, Optional
from commons import _compute_dp_matrix_2d, Match
from dataclasses import dataclass


@dataclass
class RelaxedMatch(Match):
    """Extends Match to include edit count information."""
    edits: int = 0

    def __repr__(self):
        return (
            f"RelaxedMatch(idx={self.idx}, "
            f"length={self.length}, "
            f"edits={self.edits}, "
            f"s1[{self.start_pos1}:{self.end_pos1}], "
            f"s2[{self.start_pos2}:{self.end_pos2}]), "
            f"values={self.values})"
        )


@jit(nopython=True)
def _compute_relaxed_dp(
    s1: np.ndarray, s2: np.ndarray, max_edits: int = 1
) -> np.ndarray:
    """
    Compute DP matrix allowing for edits:
    - Substitution: replace one token with another
    - Deletion: remove a token from s1
    - Insertion: add a token to s1 (remove from s2)
    - Transposition: swap adjacent tokens
    
    Returns dp[num_edits, i, j] representing match ending at i,j
    """
    m, n = len(s1), len(s2)

    # Initialize dp matrix with exact matches
    dp = np.zeros((max_edits + 1, m + 1, n + 1), dtype=np.int32)
    dp[0] = _compute_dp_matrix_2d(s1, s2)

    # Build matches with edits
    for e in range(1, max_edits + 1):
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    # Exact match - extend from same or previous edit level
                    dp[e, i, j] = max(
                        dp[e, i-1, j-1] + 1,     # Extend current match
                        dp[e-1, i-1, j-1] + 1    # Start from previous edit level
                    )
                else:
                    # Try all possible edit operations and take max:
                    
                    # 1. Substitution
                    curr_len = dp[e-1, i-1, j-1] + 1
                    
                    # 2. Deletion (skip char in s1)
                    curr_len = max(curr_len, dp[e-1, i-1, j] + 1)
                    
                    # 3. Insertion (skip char in s2)
                    curr_len = max(curr_len, dp[e-1, i, j-1] + 1)
                    
                    # 4. Transposition
                    if (
                        i > 1
                        and j > 1
                        and s1[i-1] == s2[j-2]
                        and s1[i-2] == s2[j-1]
                    ):
                        curr_len = max(curr_len, dp[e-1, i-2, j-2] + 2)
                    
                    dp[e, i, j] = curr_len

    return dp


@jit(nopython=True)
def _find_potential_relaxed_matches(
    dp: np.ndarray, min_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all potential matches from the relaxed DP matrix.
    Returns arrays of lengths, positions, and edit counts for matches.
    """
    max_edits, m, n = dp.shape
    max_matches = (m - 1) * (n - 1) * max_edits

    lengths = np.zeros(max_matches, dtype=np.int32)
    starts1 = np.zeros(max_matches, dtype=np.int32)
    ends1 = np.zeros(max_matches, dtype=np.int32)
    starts2 = np.zeros(max_matches, dtype=np.int32)
    edit_counts = np.zeros(max_matches, dtype=np.int32)

    match_idx = 0
    for e in range(max_edits):
        for i in range(1, m):
            for j in range(1, n):
                length = dp[e, i, j]
                if length >= min_length:
                    # For non-exact matches (e>0), verify they're genuine
                    if e == 0 or length > dp[0, i, j]:
                        lengths[match_idx] = length
                        starts1[match_idx] = i - length
                        ends1[match_idx] = i
                        starts2[match_idx] = j - length
                        edit_counts[match_idx] = e
                        match_idx += 1

    return (
        lengths[:match_idx],
        starts1[:match_idx],
        ends1[:match_idx],
        starts2[:match_idx],
        edit_counts[:match_idx],
    )