import torch
import numpy as np
import typing
import pathlib

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd import MMD
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
    n_run_test = 1

    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)
    mmd_estimator = MMD(kernel_function_obj=kernel_function, device_obj=device_obj, scales=init_scale)

    p_values = []
    np.random.seed(seed=1)

    for test_trial in range(0, n_run_test):
        x = np.random.normal(3, 0.5, size=(500, 2))
        y = np.random.normal(3, 0.5, size=(500, 2))
        dataset_train = TwoSampleDataSet(x, y, device_obj)

        permutation_tester = PermutationTest(is_normalize=True, mmd_estimator=mmd_estimator, dataset=dataset_train)
        statistics = permutation_tester.compute_statistic()
        threshold = permutation_tester.compute_threshold(alpha=0.05)
        p_value = permutation_tester.compute_p_value(statistics)
        logger.info(f'statistics: {statistics}, threshold: {threshold}, p-value: {p_value}')
        p_values.append(p_value)
    # end for
    ratio_same_distribution = len([p for p in p_values if p > 0.05]) / len(p_values)
    assert ratio_same_distribution > 0.7, 'Something strange in Permutation test.'


def test_all_same_value():
    """Test of an extreme case where MMD-statistics is always the same value.
    In the case, p-value should be 1.0"""
    init_scale = torch.tensor(np.array([0.05] * 5))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)
    mmd_estimator = MMD(kernel_function_obj=kernel_function, device_obj=device_obj, scales=init_scale)
    x = np.full((100, 5), 15)
    dataset_train = TwoSampleDataSet(x, x, device_obj)
    test_operator = PermutationTest(mmd_estimator=mmd_estimator, dataset=dataset_train)
    mmd_value = test_operator.compute_statistic()
    p_value = test_operator.compute_p_value(mmd_value)
    assert p_value == 1.0, f'{p_value}'


def test_statistic_greater_than_permutations():
    """Test case that statistic is greater than all values from permutation test.
    Note that the test is for an extreme case."""
    init_scale = torch.tensor(np.array([0.05] * 5))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)
    mmd_estimator = MMD(kernel_function_obj=kernel_function, device_obj=device_obj, scales=init_scale)
    x = np.full((100, 5), 15)
    dataset_train = TwoSampleDataSet(x, x, device_obj)
    test_operator = PermutationTest(mmd_estimator=mmd_estimator, dataset=dataset_train)
    p_value = test_operator.compute_p_value(np.array([100000000.0]))
    assert p_value == 0.0, f'{p_value}'


def test_fixed_random_seed():
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)
    mmd_estimator = MMD(kernel_function_obj=kernel_function, device_obj=device_obj, scales=init_scale)
    np.random.seed(seed=1)
    x = np.random.normal(3, 0.5, size=(500, 2))
    y = np.random.normal(5, 5.5, size=(500, 2))
    dataset_train = TwoSampleDataSet(x, y, device_obj)
    test_operator = PermutationTest(mmd_estimator=mmd_estimator, dataset=dataset_train)
    result_permutations = test_operator.sample_null()
    print(result_permutations)


if __name__ == '__main__':
    test_fixed_random_seed()
    test_basic_permutation_test(pathlib.Path('../../resources'))
    test_statistic_greater_than_permutations()
    test_all_same_value()

