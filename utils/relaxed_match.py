
import numpy as np
from numba import njit
from typing import List, Tuple
from dataclasses import dataclass
from numba.experimental import jitclass
from numba import int32, float64


# Define the specification for the RelaxedMatch structure
spec = [
    ("length", int32),
    ("edits", int32),
    ("start_pos1", int32),
    ("end_pos1", int32),
    ("start_pos2", int32),
    ("end_pos2", int32),
    ("values1", int32[:]),
    ("values2", int32[:]),
]


@jitclass(spec)
class RelaxedMatch:
    def __init__(
        self,
        length,
        edits,
        start_pos1,
        end_pos1,
        start_pos2,
        end_pos2,
        values1,
        values2,
    ):
        self.length = length
        self.edits = edits
        self.start_pos1 = start_pos1
        self.end_pos1 = end_pos1
        self.start_pos2 = start_pos2
        self.end_pos2 = end_pos2
        self.values1 = values1
        self.values2 = values2


@njit
def create_relaxed_match(
    length, edits, start_pos1, end_pos1, start_pos2, end_pos2, values1, values2
):
    """Factory function to create RelaxedMatch instances."""
    v1 = np.array(values1, dtype=np.int32)
    v2 = np.array(values2, dtype=np.int32)
    return RelaxedMatch(
        length, edits, start_pos1, end_pos1, start_pos2, end_pos2, v1, v2
    )


@njit
def _dp_edit_distance(s1: np.ndarray, s2: np.ndarray, max_edits: int) -> np.ndarray:
    """
    Compute edit distance with Ukkonen's band optimization.
    Only computes cells within diagonal band of width 2*max_edits+1.
    Assumes len difference check has already been done before calling.
    """
    m, n = len(s1), len(s2)
    dp = np.full((m + 1, n + 1), max_edits + 1, dtype=np.int32)
    
    # Initialize base cases
    dp[0, 0] = 0
    for i in range(1, min(m + 1, max_edits + 1)):
        dp[i, 0] = i
    for j in range(1, min(n + 1, max_edits + 1)):
        dp[0, j] = j
    
    # Fill DP table within band
    for i in range(1, m + 1):
        j_start = max(1, i - max_edits)
        j_end = min(n + 1, i + max_edits + 1)
        
        for j in range(j_start, j_end):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(
                    dp[i - 1, j] + 1,
                    dp[i, j - 1] + 1,
                    dp[i - 1, j - 1] + 1
                )
                # Check for transposition
                if (i > 1 and j > 1 and 
                    s1[i - 1] == s2[j - 2] and 
                    s1[i - 2] == s2[j - 1]):
                    dp[i, j] = min(dp[i, j], dp[i - 2, j - 2] + 1)
    
    return dp


@njit
def _find_match_with_edits(
    s1: np.ndarray, s2: np.ndarray, max_edits: int
) -> Tuple[int, int]:
    """Find the longest match with up to max_edits edits."""
    dp = _dp_edit_distance(s1, s2, max_edits)
    m, n = len(s1), len(s2)
    edits = dp[m, n]
    if edits > max_edits:
        return 0, max_edits + 1
    return max(m, n), edits


@njit
def find_all_potential_matches(
    s1: np.ndarray, s2: np.ndarray, min_length: int = 3, max_edits: int = 1, min_consecutive: int = 2
) -> list:
    """Find all potential relaxed matches between s1 and s2."""
    matches = []
    m, n = len(s1), len(s2)  # Get lengths of input sequences

    # Calculate minimum possible window size that could contain a valid match
    min_window = max(min_length, 2 * min_consecutive + max_edits)

    # Try all possible window sizes from min_length up to max sequence length
    for window in range(min_window, max(m, n) + 1):
        # Slide window over first sequence
        for i in range(m - min_window + 1):
            # Slide window over second sequence
            for j in range(n - min_window + 1):
                # Calculate end positions, capped by sequence lengths
                end1 = min(i + window, m)
                end2 = min(j + window, n)
                # Extract subsequences to compare
                seq1 = s1[i:end1]
                seq2 = s2[j:end2]

                # Skip if length difference exceeds max allowed edits
                if abs(len(seq1) - len(seq2)) > max_edits:
                    continue

                # Find length of match and number of edits needed
                match_length, num_edits = _find_match_with_edits(seq1, seq2, max_edits)

                # If match meets minimum length and edit criteria
                if match_length >= min_length and num_edits <= max_edits:
                    # Create and store match object
                    match = create_relaxed_match(
                        match_length,
                        num_edits, 
                        i,
                        end1,
                        j, 
                        end2,
                        s1[i:end1],
                        s2[j:end2],
                    )
                    matches.append(match)

    return matches


@njit
def filter_contained_matches(matches: list) -> list:
    """Filter out matches that are contained within longer matches with fewer or equal edits."""
    n = len(matches)

    # Bubble sort to order matches by length (descending)
    # We use bubble sort since sorted() isn't supported in numba
    for i in range(n):
        for j in range(0, n - i - 1):
            if matches[j].length < matches[j + 1].length:
                matches[j], matches[j + 1] = matches[j + 1], matches[j]

    final_matches = []
    # Iterate through each match to check if it's contained in a longer match
    for match in matches:
        is_contained = False
        # Compare against previously accepted matches
        for longer_match in final_matches:
            # Check if current match is fully contained within longer_match:
            # 1. Start position in sequence 1 is after longer match's start
            # 2. End position in sequence 1 is before longer match's end
            # 3. Start position in sequence 2 is after longer match's start
            # 4. End position in sequence 2 is before longer match's end
            # 5. Current match has same or more edits than longer match
            if (
                match.start_pos1 >= longer_match.start_pos1
                and match.end_pos1 <= longer_match.end_pos1
                and match.start_pos2 >= longer_match.start_pos2
                and match.end_pos2 <= longer_match.end_pos2
                and match.edits >= longer_match.edits
            ):
                is_contained = True
                break

        # Add match to final list if it's not contained in any longer match
        if not is_contained:
            final_matches.append(match)

    return final_matches


@njit
def find_relaxed_matches(
    s1: np.ndarray, s2: np.ndarray, min_length: int = 3, max_edits: int = 1
) -> list:
    """Find all relaxed matches between s1 and s2."""
    potential_matches = find_all_potential_matches(s1, s2, min_length, max_edits)
    return filter_contained_matches(potential_matches)


if __name__ == "__main__":
    from datasets import load_dataset
    from time import time

    s1 = [1, 2, 4, 3, 5, 6, 7, 11, 9, 10, 12, 13, 15, 14, 16, 17, 18, 20, 19]
    s2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20]

    # First run (includes compilation time)
    start = time()
    result = find_relaxed_matches(s1, s2, min_length=10, max_edits=4)
    print(f"First run (warmup with compilation): {time() - start:.2f}s")

    # Load the JSONL file
    dataset = load_dataset(
        "json",
        # data_files="/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Standard_GBS_120_EPOCH_75/step=2400-consumed=288000/rank0.jsonl",  
        data_files='/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Goldfish_K_5_H_13_GBS_120_EPOCH_93/step=6900-consumed=828000/rank0.jsonl',
        split="train",
    )

    s1 = dataset[2]["true_suffix"][:400]
    s2 = dataset[2]["generated_suffix"][:400]

    # Second run (actual runtime)
    start = time()
    result = find_relaxed_matches(s1, s2, min_length=4, max_edits=2)
    end = time()
    print(f"Second run (actual runtime): {end - start:.2f}s")

    # Print results
    for match in result:
        print(
            f"Match(length={match.length}, edits={match.edits}, "
            f"s1[{match.start_pos1}:{match.end_pos1}]={match.values1}, "
            f"s2[{match.start_pos2}:{match.end_pos2}]={match.values2})"
        )
