import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from numba import jit, prange
from numpy.lib.stride_tricks import sliding_window_view

@jit(nopython=True)
def compute_single_distance(s1, s2):
   """Compute Damerau-Levenshtein distance for a single pair of sequences."""
   len1, len2 = len(s1), len(s2)
   matrix = np.zeros((len1 + 1, len2 + 1))
   
   # Initialize first row and column
   matrix[0] = np.arange(len2 + 1)
   matrix[:, 0] = np.arange(len1 + 1)
   
   for i1 in range(1, len1 + 1):
       for i2 in range(1, len2 + 1):
           cost = 0 if s1[i1-1] == s2[i2-1] else 1
           
           # Regular Levenshtein operations
           matrix[i1, i2] = min(
               matrix[i1-1, i2] + 1,
               matrix[i1, i2-1] + 1,
               matrix[i1-1, i2-1] + cost
           )
           
           # Damerau transposition
           if (i1 > 1 and i2 > 1 and 
               s1[i1-1] == s2[i2-2] and 
               s1[i1-2] == s2[i2-1]):
               matrix[i1, i2] = min(
                   matrix[i1, i2],
                   matrix[i1-2, i2-2] + cost
               )
   
   return matrix[len1, len2]

@jit(nopython=True, parallel=True)
def compute_batch_distances(s1_3d, s2_3d):
   """Compute distances for all sequences in parallel."""
   batch_size, num_ngrams, _ = s1_3d.shape
   distances = np.zeros((batch_size, num_ngrams))
   
   for i in prange(batch_size):
       for j in range(num_ngrams):
           distances[i, j] = compute_single_distance(s1_3d[i, j], s2_3d[i, j])
   
   return distances

def batch_damerau_levenshtein_3d(s1_3d, s2_3d):
   """
   Optimized Damerau-Levenshtein distance calculation using Numba.
   """
   # Ensure input arrays are contiguous
   s1_3d = np.ascontiguousarray(s1_3d)
   s2_3d = np.ascontiguousarray(s2_3d)
   
   # Compute distances using parallel numba function
   distances = compute_batch_distances(s1_3d, s2_3d)
   
   # Normalize distances (not jitted since numpy operations are already fast)
   max_lengths = np.maximum(
       np.sum(s1_3d != 0, axis=2),
       np.sum(s2_3d != 0, axis=2)
   )
   max_lengths = np.maximum(max_lengths, 1)
   normalized_distances = distances / max_lengths
   
   return distances, normalized_distances

def compute_ngram_distance_damerau_levenshtein(data, N):
   """Compute n-gram distances with performance tracking."""
   import time
   
   print("Converting data to numpy arrays...")
   ref = np.array(data['true_suffix'])
   pred = np.array(data['generated_suffix'])
   
   print("Creating sliding windows...")
   ref_ngrams = sliding_window_view(ref, N, axis=1)
   pred_ngrams = sliding_window_view(pred, N, axis=1)
   
   print(f"Input shapes: ref_ngrams={ref_ngrams.shape}, pred_ngrams={pred_ngrams.shape}")
   
   # Warm up JIT compilation
   print("Warming up Numba JIT...")
   small_test = ref_ngrams[:2, :2]
   _ = compute_batch_distances(small_test, small_test)
   
   # Time the actual computation
   print("Computing distances...")
   start = time.time()
   distances, normalized = batch_damerau_levenshtein_3d(ref_ngrams, pred_ngrams)
   end = time.time()
   
   print(f"Time taken: {end - start:.2f} seconds")
   print(f"Output shapes: distances={distances.shape}, normalized={normalized.shape}")
   
   return distances, normalized

# Example usage
if __name__ == "__main__":
   file_path = '/mloscratch/homes/yixuan/PDM/inference/sparse_gutenberg_K_50_H_13/rep_128/rank0.jsonl'
   data = load_dataset('json', data_files=file_path, split='train')
   
   distances, normalized = compute_ngram_distance_damerau_levenshtein(data, N=13)
   
   # Print some statistics
   print("\nResults:")
   print(f"Average distance: {distances.mean():.4f}")
   print(f"Average normalized distance: {normalized.mean():.4f}")
   print(f"Min distance: {distances.min():.4f}")
   print(f"Max distance: {distances.max():.4f}")