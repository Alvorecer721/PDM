import pytest
import numpy as np
from datasets import load_dataset
import os
from difflib import SequenceMatcher
from transformers import AutoTokenizer


def create_dataset_config():
    """Create dataset configuration dictionary"""
    base_path = "/iopsstor/scratch/cscs/xyixuan/dataset"
    # reps = [128, 256, 512, 1024, 2048]
    reps = [1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128]
    bucket_size = 500
    
    configs = {}
    for _, rep_times in enumerate(reps):
        configs[rep_times] = {
            'GPTDataset_indices': f"{base_path}/sparse_gutenberg/rep_{rep_times}/00000_tokens/cache/GPTDataset_indices",
            'memmap_path': f"{base_path}/sparse_gutenberg/rep_{rep_times}/00000_tokens.bin",
            'jsonl_path': f"{base_path}/gutenberg/rep_{rep_times}_token.jsonl",
            'total_samples': rep_times * bucket_size,
            # Add default values or None for indices
            'document_index': None,
            'sample_index': None,
            'shuffle_index': None
        }
    
    return configs

class DatasetLoader:
    """Helper class to handle dataset loading and token extraction"""
    def __init__(self, config, setup_parameters):
        self.config = config
        self.params = setup_parameters
        self._memmap_data = None
        self._jsonl_data = None

    @property
    def memmap_data(self):
        if self._memmap_data is None:
            self._memmap_data = np.memmap(
                self.config['memmap_path'],
                mode="r",
                order="C",
                dtype=self.params['token_dtype']
            )
        return self._memmap_data

    @property
    def jsonl_data(self):
        if self._jsonl_data is None:
            self._jsonl_data = load_dataset(
                'json',
                data_files=self.config['jsonl_path'],
                split='train'
            )
        return self._jsonl_data

    def get_memmap_tokens(self, sample_idx):
        """Get tokens from memmap data for given sample index"""
        offset = sample_idx * self.params['sequence_length'] * (
            np.iinfo(self.params['token_dtype']).bits / 8
        )
        tokens = np.frombuffer(
            self.memmap_data,
            dtype=self.params['token_dtype'],
            count=self.params['sequence_length'],
            offset=int(offset)
        )[1:self.params['sequence_length']]  # Skip BOS token
        return tokens

    def get_jsonl_tokens(self, sample_idx):
        """Get tokens from JSONL data for given sample index"""
        bucket_idx = int(sample_idx % self.params['bucket_size'])
        return np.array(self.jsonl_data[bucket_idx]['input_ids'])

def generate_test_indices(total_samples, num_random=100):
    """Generate test indices: systematic and random samples"""
    if total_samples <= 500:
        return [0, total_samples - 1]
    
    # Systematic samples
    indices = [0]  # First sample
    indices.extend(range(500, total_samples - 1, 500))  # Every 500th sample
    indices.append(total_samples - 1)  # Last sample

    # Random samples
    rng = np.random.default_rng()  # Use newer numpy random generator
    existing_indices = set(indices)
    possible_indices = set(range(1, total_samples - 1)) - existing_indices
    if len(possible_indices) >= num_random:
        random_indices = rng.choice(list(possible_indices), 
                                  size=num_random, 
                                  replace=False)
        existing_indices.update(random_indices)
    
    return sorted(list(existing_indices))

DATASET_CONFIGS = create_dataset_config()
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
TOKENIZER.model_max_length = 200_000

class TestDatasetConsistency:
    @pytest.fixture
    def setup_parameters(self):
        return {
            'sequence_length': 8192,
            'bucket_size': 500,
            'token_dtype': np.int32,
        }

    @pytest.mark.parametrize(
        "rep_times,config",
        [(rep, config) for rep, config in DATASET_CONFIGS.items()]
    )
    def test_file_sizes(self, setup_parameters, rep_times, config):
        """Test if file sizes match expected total samples"""
        dataset = DatasetLoader(config, setup_parameters)
        expected_size = config['total_samples'] * setup_parameters['sequence_length']
        
        assert len(dataset.memmap_data) == expected_size, \
            f"Incorrect memmap size for repetition {rep_times}"
        
    
    @pytest.mark.parametrize(
        "rep_times,config",
        [(rep, config) for rep, config in DATASET_CONFIGS.items()]
    )
    def test_individual_sequence_length(self, setup_parameters, rep_times, config):
        dataset = DatasetLoader(config, setup_parameters)
        expected_length = setup_parameters['sequence_length'] - 1
        
        # Check each sequence in the dataset
        for idx, sequence in enumerate(dataset.jsonl_data):
            sequence_length = len(sequence['input_ids'])  # Assuming 'input_ids' is the key for sequence data
            assert sequence_length == expected_length, \
                f"Sequence length mismatch for repetition {rep_times}, sequence {idx}: " \
                f"got {sequence_length}, expected {expected_length}"


    @pytest.mark.parametrize(
        "rep_times,config",
        [(rep, config) for rep, config in DATASET_CONFIGS.items()]
    )
    def test_input_ids_consistency(self, setup_parameters, rep_times, config):
        """Test if tokens match between memmap and jsonl for specific samples"""
        dataset = DatasetLoader(config, setup_parameters)
        test_indices = generate_test_indices(config['total_samples'])

        for sample_idx in test_indices:
            memmap_tokens = dataset.get_memmap_tokens(sample_idx)
            jsonl_tokens = dataset.get_jsonl_tokens(sample_idx)

            assert len(memmap_tokens) == len(jsonl_tokens), \
                f"Token lengths don't match for repetition {rep_times}, sample {sample_idx}"
            
            if not np.array_equal(memmap_tokens, jsonl_tokens):
                error_msg = f"\nMismatch found in repetition {rep_times}, sample {sample_idx}:\n"
                error_msg += "-" * 80 + "\n"
                
                s = SequenceMatcher(None, memmap_tokens, jsonl_tokens)
                for tag, i1, i2, j1, j2 in s.get_opcodes():
                    if tag == 'replace':
                        error_msg += f"Position {i1}:{i2} vs {j1}:{j2}\n"
                        error_msg += f"Token IDs: {memmap_tokens[i1:i2]} -> {jsonl_tokens[j1:j2]}\n"
                        error_msg += f"Decoded text: '{TOKENIZER.decode(memmap_tokens[i1:i2])}' -> '{TOKENIZER.decode(jsonl_tokens[j1:j2])}'\n"
                        error_msg += "-" * 80 + "\n"

                raise AssertionError(error_msg)


def get_hash_id(directory):
    # Look for files in the directory that end with 'document_index.npy'
    for file in os.listdir(directory):
        if file.endswith('-GPTDataset-train-document_index.npy'):
            # Extract the hash ID from the filename
            hash_id = file.split('-')[0]
            return hash_id
    return None


for rep in DATASET_CONFIGS:
    indices_path = DATASET_CONFIGS[rep]['GPTDataset_indices']
    hash_id = get_hash_id(indices_path)
    
    DATASET_CONFIGS[rep].update({
        'document_index': f"{indices_path}/{hash_id}-GPTDataset-train-document_index.npy",
        'sample_index': f"{indices_path}/{hash_id}-GPTDataset-train-sample_index.npy",
        'shuffle_index': f"{indices_path}/{hash_id}-GPTDataset-train-shuffle_index.npy"
    })


class TestSparseGutenbergGoldfishRun:
    @pytest.mark.parametrize(
        "rep_times,config",
        [(rep, config) for rep, config in DATASET_CONFIGS.items()]
    )
    def test_document_index_lengths(self, rep_times, config):
        """Test if document_index.npy have correct lengths matching total samples"""

        document_index   = np.load(config['document_index'])
        expected_samples = config['total_samples'] 

        # Test lengths
        assert len(document_index) == expected_samples, \
            f"Document index length {len(document_index)} doesn't match expected {expected_samples} for rep {rep_times}"

    @pytest.mark.parametrize(
        "rep_times,config",
        [(rep, config) for rep, config in DATASET_CONFIGS.items()]
    )
    def test_sample_index_lengths(self, rep_times, config):
        """Test if sample_index.npy have correct lengths matching total samples"""
        
        sample_index     = np.load(config['sample_index'])
        expected_samples = config['total_samples'] 

        assert len(sample_index) == expected_samples, \
            f"Sample index length {len(sample_index)} doesn't match expected {expected_samples} for rep {rep_times}"

    @pytest.mark.parametrize(
        "rep_times,config",
        [(rep, config) for rep, config in DATASET_CONFIGS.items()]
    )
    def test_shuffle_index_lengths(self, rep_times, config):
        """Test if shuffle_index.npy have correct lengths matching total samples"""
        
        shuffle_index    = np.load(config['shuffle_index'])
        expected_samples = config['total_samples'] - 1  # Skip the last sample due to drop last in DataLoader

        missing_indices = set(range(expected_samples)) - set(shuffle_index)
        if missing_indices:
            print(f"Rep {rep_times} missing indices:", missing_indices)

        assert len(shuffle_index) == expected_samples, \
            f"Shuffle index length {len(shuffle_index)} doesn't match expected {expected_samples} for rep {rep_times}"
