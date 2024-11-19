import numpy as np
from numba import jit
from dataclasses import dataclass

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