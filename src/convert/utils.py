import os
import shutil
import tempfile
import torch.distributed as dist

  
class TempFolder(object):
    def __init__(self, prefix='layout_test_'):
        self._path = tempfile.mkdtemp(prefix=prefix)
        print('Create Temp Folder: {}'.format(self._path))

    def get_directory(self, sub_folder='.'):
        new_path = os.path.realpath(os.path.join(self._path, sub_folder))
        if os.path.normpath(new_path) == os.path.normpath('/'):
            raise Exception('Error, path is too dangerous.')
        return new_path

    def __del__(self):
        shutil.rmtree(self._path, True)
        print('Remove Temp Folder: {}'.format(self._path))


def is_rank_0():
    """Helper function to check if current process is rank 0"""
    # Check if we're in a distributed environment
    if dist.is_initialized():
        return dist.get_rank() == 0
    # If not distributed, we're on rank 0
    return True