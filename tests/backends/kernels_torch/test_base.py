import numpy as np
import pathlib
import tempfile
import torch
from torch.utils.data import DataLoader

from model_criticism_mmd.models import datasets
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel
from model_criticism_mmd.models.static import DEFAULT_DEVICE


def test_base_delete_padding():
    """A test case where X and Y have difference length of samples"""
    tmp_work_dir = pathlib.Path(tempfile.mktemp())
    tmp_work_dir.mkdir()
    x = np.random.normal(loc=1.0, scale=15, size=(130, 1000))
    y = np.random.normal(loc=1.0, scale=15, size=(150, 1000))
    sample_dataset = datasets.TwoSampleIterDataSet(x, y, device_obj=DEFAULT_DEVICE,
                                                   working_dir=tmp_work_dir)
    loader = DataLoader(sample_dataset, batch_size=100)
    batches = list(loader)
    assert all(batches[1][0].isnan().detach().numpy().tolist())
    base_kernel = BaseKernel(device_obj=DEFAULT_DEVICE, possible_shapes=(2,))
    tensor_after_delete = base_kernel.delete_padding(batches[1][0])
    assert tensor_after_delete.shape[0] == 30


if __name__ == '__main__':
    test_base_delete_padding()
