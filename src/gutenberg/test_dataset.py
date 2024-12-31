import pytest
import numpy as np
from datasets import load_dataset

def create_dataset_config():
    """Create dataset configuration dictionary"""
    base_path = "/iopsstor/scratch/cscs/xyixuan/dataset"
    reps_ordered = [128, 16, 1, 24, 2, 32, 3, 48, 4, 64, 8, 96]
    bucket_size = 500
    
    configs = {}
    for idx, rep_times in enumerate(reps_ordered):
        bin_num = str(idx).zfill(5)
        configs[rep_times] = {
            'memmap_path': f"{base_path}/sparse_gutenberg_reptitions/{bin_num}_tokens.bin",
            'jsonl_path': f"{base_path}/gutenberg/rep_{rep_times}_token.jsonl",
            'total_samples': rep_times * bucket_size
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
    def test_input_ids_consistency(self, setup_parameters, rep_times, config):
        """Test if tokens match between memmap and jsonl for specific samples"""
        dataset = DatasetLoader(config, setup_parameters)
        test_indices = generate_test_indices(config['total_samples'])

        for sample_idx in test_indices:
            memmap_tokens = dataset.get_memmap_tokens(sample_idx)
            jsonl_tokens = dataset.get_jsonl_tokens(sample_idx)

            assert len(memmap_tokens) == len(jsonl_tokens), \
                f"Token lengths don't match for repetition {rep_times}, sample {sample_idx}"

            assert np.array_equal(memmap_tokens, jsonl_tokens), \
                f"Tokens don't match for repetition {rep_times}, sample {sample_idx}"

        
# train_document_index = np.load('GPTDataset_sparse_gutenberg/rep_16/bf4e1461f9eae904f2f55bfa4300ee10-GPTDataset-train-document_index.npy') 
# train_sample_index = np.load('GPTDataset_sparse_gutenberg/rep_16/bf4e1461f9eae904f2f55bfa4300ee10-GPTDataset-train-sample_index.npy')
# train_shuffle_index  = np.load('GPTDataset_sparse_gutenberg/rep_16/bf4e1461f9eae904f2f55bfa4300ee10-GPTDataset-train-shuffle_index.npy')