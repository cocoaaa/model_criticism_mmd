import torch

from model_criticism_mmd.models import datasets
from torch.utils.data import DataLoader
import tempfile
import pathlib
import numpy as np


def test_TwoSampleIterDataSet():
    tmp_work_dir = pathlib.Path(tempfile.mktemp())
    tmp_work_dir.mkdir()
    x = np.random.normal(loc=1.0, scale=15, size=(1000, 1000))
    y = np.random.normal(loc=1.0, scale=15, size=(1000, 1000))
    sample_dataset = datasets.TwoSampleIterDataSet(x, y, device_obj=torch.device('cpu'))
    loader = DataLoader(sample_dataset, batch_size=100)
    batches = list(loader)
    assert len(batches) == 10
    assert len(batches[0]) == 2


if __name__ == '__main__':
    test_TwoSampleIterDataSet()

