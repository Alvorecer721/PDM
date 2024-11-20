import numpy as np
from numba import jit
from enum import Enum
from typing import List, Tuple, Optional
from commons import _compute_dp_matrix_2d, Match
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


def _find_match_with_edits(
    s1: np.ndarray, s2: np.ndarray, allowed_edits: int
) -> Tuple[int, List[int]]:
    """
    Find the longest match with exactly allowed_edits edits.
    Returns the length of match and list of edit positions.
    """
    edits = 0
    i = 0
    edit_positions = []
    
    while i < len(s1) and i < len(s2):
        if s1[i] == s2[i]:
            i += 1
            continue
            
        # We've found a mismatch
        current_edit_start = i
        
        # Check for transposition
        if (i + 1 < len(s1) and i + 1 < len(s2) and 
            s1[i] == s2[i + 1] and s1[i + 1] == s2[i]):
            edits += 1
            edit_positions.append(i)
            i += 2  # Skip both transposed elements
        else:
            # Single substitution
            edits += 1
            edit_positions.append(i)
            i += 1
                
        if edits > allowed_edits:
            # If we exceed allowed edits, backtrack to last valid position
            return current_edit_start, edit_positions[:-1]
            
    return i, edit_positions


def find_relaxed_matches(
    s1: np.ndarray,
    s2: np.ndarray,
    min_length: int = 3,
    max_edits: int = 1
) -> List[RelaxedMatch]:
    """Find all relaxed matches between s1 and s2."""
    matches = []
    m, n = len(s1), len(s2)
    
    # For each edit distance
    for allowed_edits in range(max_edits + 1):
        # Try all possible starting positions
        for i in range(m):
            for j in range(n):
                seq1 = s1[i:]
                seq2 = s2[j:]
                
                match_length, edit_positions = _find_match_with_edits(seq1, seq2, allowed_edits)
                
                if match_length >= min_length and len(edit_positions) == allowed_edits:
                    matches.append(RelaxedMatch(
                        length=match_length,
                        edits=allowed_edits,
                        start_pos1=i,
                        end_pos1=i + match_length,
                        start_pos2=j,
                        end_pos2=j + match_length,
                        values1=s1[i:i + match_length],
                        values2=s2[j:j + match_length]
                    ))
    
    # Sort matches by length (descending) and edits (ascending)
    matches.sort(key=lambda m: (-m.length, m.edits))
    
    # Filter out contained matches
    final_matches = []
    for match in matches:
        # Check if this match is contained in any longer match
        is_contained = False
        for longer_match in final_matches:
            if (match.start_pos1 >= longer_match.start_pos1 and 
                match.end_pos1 <= longer_match.end_pos1):
                is_contained = True
                break
        
        if not is_contained:
            final_matches.append(match)
    
    return final_matches



if __name__ == "__main__":
    s1 = np.array([1, 2, 4, 3, 5, 6, 7, 11, 9, 10, 12, 13, 15, 14, 16, 17, 18, 20, 19])
    s2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20])

    result = find_relaxed_matches(s1, s2, max_edits=3)

    for res in result:
        print(res)