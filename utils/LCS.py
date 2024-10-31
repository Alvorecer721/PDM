from numba import jit, prange
import numpy as np
import pandas as pd
import time


@jit(nopython=True)  # Makes the function run faster using Numba compilation
def _find_lcs(s1, s2):
    """
    Find the longest common substring between two strings using dynamic programming.
    
    Parameters
    ----------
    s1 : np.ndarray
        First input string to compare
    s2 : np.ndarray
        Second input string to compare
        
    Returns
    -------
    tuple
        A 5-tuple containing:
        - start_pos1 (int): Starting position of match in first string 
        - end_pos1 (int): Ending position of match in first string (exclusive)
        - start_pos2 (int): Starting position of match in second string 
        - end_pos2 (int): Ending position of match in second string (exclusive)
        - max_length (int): Length of the longest common substring

    Example:
    --------
    >>> s1 = "ABCDEF"
    >>> s2 = "XBCDY"
    >>> start1, end1, start2, end2, length = find_longest_common_substring(s1, s2)
    >>> print(f"Match: {s1[start1:end1]}")  # Prints: Match: BCD
    >>> print(f"Length: {length}")  # Prints: Length: 3
    >>> print(f"Position in s1: {start1}-{end1}")  # Prints: Position in s1: 1-4
    >>> print(f"Position in s2: {start2}-{end2}")  # Prints: Position in s2: 1-4
    """    
    # Example:
    # s1 = "ABCDEF"
    # s2 = "XBCDY"
    
    m, n = len(s1), len(s2)  # m=6, n=5
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)  # Creates 7x6 matrix of zeros
    
    max_length = 0      # Tracks longest match found
    ending_pos1 = 0     # Where match ends in s1
    ending_pos2 = 0     # Where match ends in s2

    # Fill the dp matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:            # If characters match
                dp[i, j] = dp[i-1, j-1] + 1   # Add 1 to previous match length
                if dp[i, j] > max_length:     # If this is longest match so far
                    max_length = dp[i, j]     # Update max length
                    ending_pos1 = i           # Save ending positions
                    ending_pos2 = j
    
    # For example, if we found "BCD":
    # max_length = 3
    # ending_pos1 = 4 (after 'D' in ABCDEF)
    # ending_pos2 = 4 (after 'D' in XBCDY)
    
    # Calculate starting positions by subtracting length
    start_pos1 = ending_pos1 - max_length  # 4-3 = 1 (starts at 'B')
    start_pos2 = ending_pos2 - max_length  # 4-3 = 1 (starts at 'B')
    
    return max_length, start_pos1, ending_pos1, start_pos2, ending_pos2


def find_longest_common_substrings(reference, predicted):
    """
    Find longest common substrings for multiple pairs of strings in parallel.
    
    Parameters
    ----------
    reference : array-like
        List of tokenised reference strings to compare
    predicted : array-like
        List of tokenised predicted strings to compare

    Returns
    -------
    dict or pd.DataFrame
        For single string input: returns dictionary with keys
            'max_length', 'start_pos1', 'end_pos1', 'start_pos2', 'end_pos2'
        For list input: returns DataFrame with columns
            max_length, start_pos1, end_pos1, start_pos2, end_pos2

    Examples
    --------
    # Single string pair:
    >>> result = find_longest_common_substrings("ABCDEF", "XBCDY")
    >>> print(result)  # {'max_length': 3, 'start_pos1': 1, 'end_pos1': 4, ...}
    
    # Multiple string pairs:
    >>> reference = ["ABCDEF", "WXYZ"]
    >>> predicted = ["XBCDY", "XYZ"]
    >>> results = find_longest_common_substrings(reference, predicted)
    >>> print(type(results))  # pandas.DataFrame
    """
    reference = np.array(reference)
    predicted = np.array(predicted)

    # Check dimensions
    if reference.ndim != predicted.ndim:
        raise ValueError("reference and predicted must have same number of dimensions")
    
    # Handle single list inputs
    if reference.ndim == 1:
        results = _find_lcs(reference, predicted)
        return {
            'max_length': results[0],
            'start_pos1': results[1],
            'end_pos1': results[2],
            'start_pos2': results[3],
            'end_pos2': results[4]
        }

    # Handle list of lists inputs
    elif reference.ndim == 2:
        if len(reference) != len(predicted):
            raise ValueError("reference and predicted must have the same length")
        
        # Pre-allocate results array
        n_samples = len(reference)
        results = np.zeros((n_samples, 5), dtype=np.int32)

        # Warm up JIT compilation
        print("Warming up Numba JIT...")
        _ = _find_lcs(np.array([1, 2]), np.array([2, 3]))
        
        # Process each pair of strings and log time taken
        print(f"Processing {n_samples} sequence pairs in parallel...")
        processing_start = time.time()
        for i in prange(n_samples):
            results[i] = _find_lcs(reference[i], predicted[i])
        processing_time = time.time() - processing_start
        print(f"Time taken: {processing_time:.2f} seconds")
    
        return pd.DataFrame(
            results,
            columns=['max_length', 'start_pos1', 'end_pos1', 'start_pos2', 'end_pos2']
        )
    
    else:
        raise ValueError("Both inputs must be either lists or lists of lists")
    

# Example usage:
if __name__ == "__main__":
    ...