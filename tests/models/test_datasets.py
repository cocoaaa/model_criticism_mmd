import math

import torch

from model_criticism_mmd.models import datasets
from torch.utils.data import DataLoader
import tempfile
import pathlib
import numpy as np
from model_criticism_mmd.models.static import DEFAULT_DEVICE


def test_TwoSampleDataSet_padding():
    tmp_work_dir = pathlib.Path(tempfile.mktemp())
    tmp_work_dir.mkdir()
    # case-1 len(y) > len(x)
    x = np.random.normal(loc=1.0, scale=15, size=(100, 1000))
    y = np.random.normal(loc=1.0, scale=15, size=(150, 1000))
    sample_dataset = datasets.TwoSampleIterDataSet(x, y, device_obj=DEFAULT_DEVICE,
                                                   working_dir=tmp_work_dir)
    loader = DataLoader(sample_dataset, batch_size=100)
    batches = list(loader)
    assert len(batches) == 2
    assert len(batches[0]) == 2


def test_TwoSampleIterDataSet_padding():
    """A test case where x and y has different size of samples."""
    tmp_work_dir = pathlib.Path(tempfile.mktemp())
    tmp_work_dir.mkdir()
    # case-1 len(y) > len(x)
    x = np.random.normal(loc=1.0, scale=15, size=(100, 1000))
    y = np.random.normal(loc=1.0, scale=15, size=(150, 1000))
    sample_dataset = datasets.TwoSampleIterDataSet(x, y, device_obj=DEFAULT_DEVICE,
                                                   working_dir=tmp_work_dir)
    loader = DataLoader(sample_dataset, batch_size=100)
    batches = list(loader)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    # to check padding
    assert all(batches[1][0].isnan().detach().numpy().tolist())
    # case-2 len(x) > len(y)
    x = np.random.normal(loc=1.0, scale=15, size=(150, 1000))
    y = np.random.normal(loc=1.0, scale=15, size=(100, 1000))
    tmp_work_dir = pathlib.Path(tempfile.mktemp())
    tmp_work_dir.mkdir()
    sample_dataset = datasets.TwoSampleIterDataSet(x, y, device_obj=DEFAULT_DEVICE,
                                                   working_dir=tmp_work_dir)
    loader = DataLoader(sample_dataset, batch_size=100)
    batches = list(loader)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    # to check padding
    assert all(batches[1][0].isnan().detach().numpy().tolist())


def test_TwoSampleIterDataSet_basic():
    tmp_work_dir = pathlib.Path(tempfile.mktemp())
    tmp_work_dir.mkdir()
    x = np.random.normal(loc=1.0, scale=15, size=(1000, 1000))
    y = np.random.normal(loc=1.0, scale=15, size=(1000, 1000))
    sample_dataset = datasets.TwoSampleIterDataSet(x, y, device_obj=DEFAULT_DEVICE)
    loader = DataLoader(sample_dataset, batch_size=100)
    batches = list(loader)
    assert len(batches) == 10
    assert len(batches[0]) == 2


if __name__ == '__main__':
    test_TwoSampleIterDataSet_basic()
    test_TwoSampleIterDataSet_padding()
    test_TwoSampleDataSet_padding()
