import torch
import numpy as np
import typing
import pathlib

from model_criticism_mmd import ModelTrainerTorchBackend, ModelTrainerTheanoBackend, MMD, split_data
from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction
from model_criticism_mmd.supports.permutation_tests import PermutationTest
from model_criticism_mmd.models import TwoSampleDataSet


def data_processor(resource_path_root: pathlib.Path
                   ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']

    return x_train, y_train, x_test, y_test


def test_basic_permutation_test(resource_path_root):
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)
    mmd_estimator = MMD(kernel_function_obj=kernel_function, device_obj=device_obj, scales=init_scale)
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)

    permutation_tester = PermutationTest(mmd_estimator=mmd_estimator, dataset=dataset_train)
    statistics = permutation_tester.compute_statistic()
    threshold = permutation_tester.compute_threshold(alpha=0.05)
    p_value = permutation_tester.compute_p_value(statistics)


if __name__ == '__main__':
    test_basic_permutation_test(pathlib.Path('../resources'))
