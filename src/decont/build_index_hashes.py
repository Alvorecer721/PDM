import argparse
import shutil
import tempfile
from pathlib import Path
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.decont import NGramsDecontConfig, NGramsDecontIndexer


def parse_filename(file_path):
    """Parse filename to extract base name and chunk name."""
    full_name = file_path.name.replace('.jsonl.gz', '').replace('.jsonl', '')
    
    # Expected format: finewebedu_000003_chunk_001.jsonl.gz
    parts = full_name.split('_')
    if len(parts) >= 4 and 'chunk' in parts:
        chunk_idx = parts.index('chunk')
        base_name = '_'.join(parts[:chunk_idx])  # e.g., finewebedu_000003
        chunk_name = '_'.join(parts[chunk_idx:])  # e.g., chunk_001
    else:
        # Fallback if naming doesn't match expected pattern
        base_name = full_name
        chunk_name = full_name
    
    return base_name, chunk_name


def create_pipeline(file_path, output_folder, config, limit):
    """Create the datatrove pipeline."""

    reader_kwargs = {
        "data_folder": str(file_path.parent),
        "glob_pattern": file_path.name,
        "text_key": "text",
    }

    if limit > 0:
        reader_kwargs["limit"] = limit
    
    return [
        JsonlReader(**reader_kwargs),
        NGramsDecontIndexer(
            output_folder=str(output_folder),
            config=config,
            lighteval_tasks=[],
            custom_lighteval_tasks=None
        )
    ]


def move_output_files(temp_dir, chunk_folder):
    """Move output files from temp directory to final location."""
    temp_path = Path(temp_dir)
    
    # Move stats file
    stats_file = temp_path / "stats.json"
    final_stats_file = chunk_folder / "stats.json"
    if stats_file.exists():
        shutil.move(str(stats_file), str(final_stats_file))
    
    # Move hash file
    hash_file = temp_path / "input.index.hashes"
    final_hash_file = chunk_folder / "input.index.hashes"
    if hash_file.exists():
        shutil.move(str(hash_file), str(final_hash_file))


def build_index(file_path, output_dir, n_gram_size=13, limit=-1):
    """Build n-gram index for decontamination."""
    file_path = Path(file_path)
    base_name, chunk_name = parse_filename(file_path)
    
    # Create folder structure with separate folder for each chunk
    base_folder = Path(output_dir) / base_name
    chunk_folder = base_folder / chunk_name
    chunk_folder.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = NGramsDecontConfig(
        n_grams=n_gram_size,
        find_query_ngrams=True,
        find_overlap_ngrams=False
    )
    
    # Create unique temporary directory to avoid conflicts
    temp_dir = tempfile.mkdtemp(prefix=f"{chunk_name}_", dir=chunk_folder)
    
    try:
        # Create and run pipeline
        pipeline = create_pipeline(file_path, chunk_folder, config, limit)
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=1,
            logging_dir=temp_dir
        )
        executor.run()
        
        # Move output files to final location
        move_output_files(temp_dir, chunk_folder)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return str(chunk_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--output-dir', default='/iopsstor/scratch/cscs/xyixuan/PDM/results/decont')
    parser.add_argument('--n-gram-size', type=int, default=13)
    parser.add_argument('--limit', type=int, default=-1, help='Number of documents to process (-1 for all)')
    
    args = parser.parse_args()
    build_index(args.file, args.output_dir, args.n_gram_size, args.limit)


if __name__ == "__main__":
    main()