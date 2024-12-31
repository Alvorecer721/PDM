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
            'GPTDataset_indices': f"{base_path}/sparse_gutenberg_reptitions/{bin_num}_tokens/cache/GPTDataset_indices",
            'memmap_path': f"{base_path}/sparse_gutenberg_reptitions/{bin_num}_tokens.bin",
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
            
hash_ids = {
    128: '8469a75be5e96b8174b5727041f66581',
    16: 'bf4e1461f9eae904f2f55bfa4300ee10',
    1: '156e236bed1bce08ee5b3267c54c8146',
    24: '7d4bdf7279b4bfd6a7211021ecc9abec',
    2: '72aa6350afc32b6e5c1ec6db1d6d4633',
    32: '5dae9d02153c916f4c625d456880aa0b',
    3: '12b6f383aeffe0702700fa7c5e5229c2',
    48: '5419b9e1a395b9424591206269dccac1',
    4: '1790918d5e4ec2af7934037985dbc7f2',
    64: '9639a884e708b97d499da4f598fe79b6',
    8: '4edd082f0f4a2eadab42fe79bcf16732',
    96: '1d8fe0ae55a30d2eee50620a378a4afd'
}

for rep, hash_id in hash_ids.items():
    DATASET_CONFIGS[rep].update({
        'document_index': f"{DATASET_CONFIGS[rep]['GPTDataset_indices']}/{hash_id}-GPTDataset-train-document_index.npy",
        'sample_index': f"{DATASET_CONFIGS[rep]['GPTDataset_indices']}/{hash_id}-GPTDataset-train-sample_index.npy",
        'shuffle_index': f"{DATASET_CONFIGS[rep]['GPTDataset_indices']}/{hash_id}-GPTDataset-train-shuffle_index.npy"
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
        expected_samples = config['total_samples'] 

        missing_indices = set(range(expected_samples)) - set(shuffle_index)
        if missing_indices:
            print(f"Rep {rep_times} missing indices:", missing_indices)

        assert len(shuffle_index) == expected_samples, \
            f"Shuffle index length {len(shuffle_index)} doesn't match expected {expected_samples} for rep {rep_times}"
