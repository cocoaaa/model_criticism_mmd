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


if __name__ == '__main__':
    test_basic_permutation_test(pathlib.Path('../../resources'))
