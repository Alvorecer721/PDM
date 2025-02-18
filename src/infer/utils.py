import torch
import numpy as np
import random
import torch.distributed as dist


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def is_rank_0():
    """Helper function to check if current process is rank 0"""
    # Check if we're in a distributed environment
    if dist.is_initialized():
        return dist.get_rank() == 0
    # If not distributed, we're on rank 0
    return True