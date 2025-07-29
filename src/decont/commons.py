"""
Common utilities for decontamination analysis.
"""

import struct
from pathlib import Path
import numpy as np
import tempfile
import shutil
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontIndexer


def load_hash_index(index_path):
    """Load n-gram hashes from index file (DataTrove binary format).
    
    Args:
        index_path: Path to the .index.hashes file
        
    Returns:
        Dict with 'hashes' key containing the hash values
    """
    with open(index_path, 'rb') as f:
        data = f.read()
        # DataTrove stores hashes as 64-bit unsigned integers
        num_hashes = len(data) // 8
        hashes = struct.unpack(f'{num_hashes}Q', data)
        # Return in the expected format
        return {'hashes': hashes}


def calculate_contamination_ratio(source_hashes, target_hashes):
    """Calculate contamination ratio between two hash sets.
    
    Measures what percentage of source n-grams appear in target.
    
    Args:
        source_hashes: Dict with 'hashes' key (what we're checking for contamination)
        target_hashes: Dict with 'hashes' key (reference to check against)
        
    Returns:
        Dict with contamination statistics
    """
    # Convert to sets for efficient intersection
    source_set = set(source_hashes.get('hashes', []))
    target_set = set(target_hashes.get('hashes', []))
    
    # Calculate intersection
    matching_hashes = source_set.intersection(target_set)
    
    # Calculate contamination ratio
    total_source_ngrams = len(source_set)
    matching_ngrams = len(matching_hashes)
    contamination_ratio = matching_ngrams / total_source_ngrams if total_source_ngrams > 0 else 0
    
    return {
        'contamination_ratio': contamination_ratio,
        'matching_ngrams': matching_ngrams,
        'total_source_ngrams': total_source_ngrams,
        'total_target_ngrams': len(target_set)
    }


def find_matching_indices(source_dir, target_dir, source_pattern="*", target_pattern="*"):
    """Find matching index files between two directories.
    
    This is a generic function that can match indices based on flexible patterns.
    
    Args:
        source_dir: Path to source indices directory
        target_dir: Path to target indices directory  
        source_pattern: Glob pattern for source directories
        target_pattern: Glob pattern for target directories
        
    Returns:
        List of tuples: [(identifier, source_path, target_path), ...]
    """
    source_indices = {}
    target_indices = {}
    
    # Scan source indices
    for index_dir in Path(source_dir).glob(source_pattern):
        if index_dir.is_dir():
            index_file = index_dir / "input.index.hashes"
            if index_file.exists():
                source_indices[index_dir.name] = index_file
    
    # Scan target indices
    for index_dir in Path(target_dir).glob(target_pattern):
        if index_dir.is_dir():
            index_file = index_dir / "input.index.hashes"
            if index_file.exists():
                target_indices[index_dir.name] = index_file
    
    # Find matches based on directory names
    # This allows flexible matching strategies
    matches = []
    for source_name, source_path in source_indices.items():
        # You can customize the matching logic here
        # For now, we'll do exact name matching
        if source_name in target_indices:
            matches.append((source_name, source_path, target_indices[source_name]))
    
    return matches


def find_repetition_based_matches(source_dir, target_dir, source_pattern="rep_*", target_pattern="rep_*"):
    """Find matching indices based on repetition numbers.
    
    Specialized function for repetition-based directory structures.
    
    Args:
        source_dir: Path to source indices directory
        target_dir: Path to target indices directory
        source_pattern: Glob pattern for source rep directories
        target_pattern: Glob pattern for target rep directories
        
    Returns:
        Tuple of (source_reps dict, target_reps dict, common_reps list)
    """
    source_reps = {}
    target_reps = {}
    
    # Extract repetition number from directory name
    def extract_rep_num(dir_name):
        parts = dir_name.split('_')
        for i, part in enumerate(parts):
            if part == 'rep' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
        return None
    
    # Scan source indices
    for rep_dir in Path(source_dir).glob(source_pattern):
        if rep_dir.is_dir():
            rep_num = extract_rep_num(rep_dir.name)
            if rep_num is not None:
                index_file = rep_dir / "input.index.hashes"
                if index_file.exists():
                    source_reps[rep_num] = index_file
    
    # Scan target indices
    for rep_dir in Path(target_dir).glob(target_pattern):
        if rep_dir.is_dir():
            rep_num = extract_rep_num(rep_dir.name)
            if rep_num is not None:
                index_file = rep_dir / "input.index.hashes"
                if index_file.exists():
                    target_reps[rep_num] = index_file
    
    # Find common repetitions
    common_reps = sorted(set(source_reps.keys()) & set(target_reps.keys()))
    
    return source_reps, target_reps, common_reps


def print_contamination_summary(results, source_name="Source", target_name="Target"):
    """Print a formatted summary of contamination results.
    
    Args:
        results: Dict with contamination results by identifier
        source_name: Name to use for source dataset
        target_name: Name to use for target dataset
    """
    if not results:
        print("\nNo results to display")
        return
    
    print("\n" + "="*80)
    print(f"{source_name} → {target_name} Contamination Results")
    print("="*80)
    print(f"{'Identifier':<20} {'Contamination %':<20} {'Matching n-grams':<20} {f'{source_name} n-grams':<20}")
    print("="*80)
    
    # Sort results by identifier
    sorted_results = sorted(results.items())
    
    for identifier, data in sorted_results:
        contamination_pct = data['contamination_ratio'] * 100
        matching = data['matching_ngrams']
        total_source = data.get('total_source_ngrams', data.get('total_inference_ngrams', 0))
        
        print(f"{identifier:<20} {contamination_pct:>6.2f}%              {matching:<20,} {total_source:<20,}")
    
    print("="*80)
    
    # Calculate and print statistics
    contamination_values = [d['contamination_ratio'] * 100 for d in results.values()]
    print(f"\nStatistics:")
    print(f"  Average contamination: {np.mean(contamination_values):.2f}%")
    print(f"  Min contamination: {np.min(contamination_values):.2f}%")
    print(f"  Max contamination: {np.max(contamination_values):.2f}%")
    print(f"  Std deviation: {np.std(contamination_values):.2f}%")


def create_index_pipeline(file_path, output_folder, n_gram_size=13, text_key="text", limit=-1):
    """Create DataTrove pipeline for n-gram indexing.
    
    Args:
        file_path: Path to the JSONL file to index
        output_folder: Directory where index will be saved
        n_gram_size: Size of n-grams to index (default: 13)
        text_key: Key in JSON containing text to index (default: "text")
        limit: Number of documents to process (-1 for all)
        
    Returns:
        Pipeline list ready for execution
    """
    file_path = Path(file_path)
    
    # Configuration
    config = NGramsDecontConfig(
        n_grams=n_gram_size,
        find_query_ngrams=True,
        find_overlap_ngrams=False
    )
    
    # Create reader with optional limit
    reader_kwargs = {
        "data_folder": str(file_path.parent),
        "glob_pattern": file_path.name,
        "text_key": text_key,
    }
    
    if limit > 0:
        reader_kwargs["limit"] = limit
    
    # Create pipeline
    pipeline = [
        JsonlReader(**reader_kwargs),
        NGramsDecontIndexer(
            output_folder=str(output_folder),
            config=config,
            lighteval_tasks=[],
            custom_lighteval_tasks=None
        )
    ]
    
    return pipeline


def build_index_for_file(file_path, output_dir, n_gram_size=13, text_key="text", limit=-1, use_subdirectory=True):
    """Build n-gram index for a single file using DataTrove.
    
    Args:
        file_path: Path to the JSONL file to index
        output_dir: Directory where index will be saved
        n_gram_size: Size of n-grams to index (default: 13)
        text_key: Key in JSON containing text to index (default: "text")
        limit: Number of documents to process (-1 for all)
        use_subdirectory: If True, create a subdirectory based on filename (default: True)
        
    Returns:
        Path to the output folder containing the index
    """
    file_path = Path(file_path)
    
    # Create output directory based on file name
    if use_subdirectory:
        output_folder = Path(output_dir) / file_path.stem
    else:
        output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create unique temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"{file_path.stem}_", dir=output_folder)
    
    try:
        # Create pipeline using the new function
        pipeline = create_index_pipeline(file_path, output_folder, n_gram_size, text_key, limit)
        
        # Run pipeline - NGramsDecontIndexer requires single task
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=1,  # NGramsDecontIndexer only supports single worker
            logging_dir=temp_dir
        )
        executor.run()
        
        # Move output files from temp to final location
        stats_file = Path(temp_dir) / "stats.json"
        hash_file = Path(temp_dir) / "input.index.hashes"
        
        if stats_file.exists():
            shutil.move(str(stats_file), str(output_folder / "stats.json"))
        if hash_file.exists():
            shutil.move(str(hash_file), str(output_folder / "input.index.hashes"))
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return str(output_folder)


def build_index_wrapper(args_dict):
    """Wrapper function for parallel index building.
    
    Args:
        args_dict: Dictionary with keys:
            - file_path: Path to the file
            - output_dir: Output directory
            - n_gram_size: N-gram size (default: 13)
            - text_key: Text key (default: "text")
            - overwrite: Whether to overwrite existing (default: False)
            - limit: Document limit (default: -1)
        
    Returns:
        Status message string
    """
    file_path = args_dict['file_path']
    output_dir = args_dict['output_dir']
    n_gram_size = args_dict.get('n_gram_size', 13)
    text_key = args_dict.get('text_key', 'text')
    overwrite = args_dict.get('overwrite', False)
    limit = args_dict.get('limit', -1)
    
    try:
        # Check if index already exists
        output_folder = output_dir / file_path.stem
        hash_file = output_folder / "input.index.hashes"
        stats_file = output_folder / "stats.json"
        
        if hash_file.exists() and stats_file.exists() and not overwrite:
            return f"Skipped: {file_path.name} - index already exists"
        
        build_index_for_file(file_path, output_dir, n_gram_size, text_key, limit)
        return f"Success: {file_path.name}"
    except Exception as e:
        return f"Error: {file_path.name} - {e}"