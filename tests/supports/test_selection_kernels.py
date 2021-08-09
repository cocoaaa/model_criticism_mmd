import torch

from model_criticism_mmd.supports.selection_kernels import SelectionKernels
from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction, MaternKernelFunction
from model_criticism_mmd.models.datasets import TwoSampleDataSet

import numpy as np


def test_selection_kernels_without_training():
    device_obj = torch.device('cpu')
    x_train = np.random.normal(3, 0.5, size=(500, 2))
    y_train = np.random.normal(3, 0.5, size=(500, 2))
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)

    x_test = np.random.normal(3, 0.5, size=(100, 2))
    y_test = np.random.normal(3, 0.5, size=(100, 2))
    dataset_validation = TwoSampleDataSet(x_test, y_test, device_obj)

    scales = torch.tensor([0.05, 0.05])
    kernels = [(scales, BasicRBFKernelFunction(device_obj=device_obj)),
               (scales, MaternKernelFunction(nu=0.5, device_obj=device_obj))]
    selection_obj = SelectionKernels(candidate_kernels=kernels, dataset_validation=dataset_validation)
    result_select = selection_obj.run_selection()
    assert result_select[0].trained_mmd_parameter is None


def test_selection_kernels_with_training():
    device_obj = torch.device('cpu')
    x_train = np.random.normal(3, 0.5, size=(500, 2))
    y_train = np.random.normal(3, 0.5, size=(500, 2))
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)

    x_test = np.random.normal(3, 0.5, size=(100, 2))
    y_test = np.random.normal(3, 0.5, size=(100, 2))
    dataset_validation = TwoSampleDataSet(x_test, y_test, device_obj)

    # with the given scales
    scales = torch.tensor([0.05, 0.05])
    kernels = [(scales, BasicRBFKernelFunction(device_obj=device_obj)),
               (scales, MaternKernelFunction(nu=0.5, device_obj=device_obj))]
    selection_obj = SelectionKernels(candidate_kernels=kernels,
                                     dataset_validation=dataset_validation,
                                     dataset_training=dataset_train,
                                     is_training=True,
                                     num_epochs=100)
    result_select = selection_obj.run_selection()
    assert result_select[0].trained_mmd_parameter is not None
    # without any given scales
    kernels = [(None, BasicRBFKernelFunction(device_obj=device_obj)),
               (None, MaternKernelFunction(nu=0.5, device_obj=device_obj))]
    selection_obj = SelectionKernels(candidate_kernels=kernels,
                                     dataset_validation=dataset_validation,
                                     dataset_training=dataset_train,
                                     is_training=True,
                                     num_epochs=100)
    result_select = selection_obj.run_selection()
    assert result_select[0].trained_mmd_parameter is not None


if __name__ == '__main__':
    test_selection_kernels_with_training()
    test_selection_kernels_without_training()
