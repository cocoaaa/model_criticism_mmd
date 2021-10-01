import pathlib
import pickle
import torch

from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction
from model_criticism_mmd import TestResultGroupsFormatter, StatsTestEvaluator
import numpy as np
from tempfile import mkdtemp

def test_unit():
    x_same = np.random.normal(loc=5, scale=0.5, size=(100, 200))
    seq_x_eval = [np.random.normal(loc=5, scale=0.5, size=(100, 200)) for i in range(0, 3)]
    y_same = np.random.normal(loc=5, scale=0.5, size=(100, 200))
    seq_y_eval_same = [np.random.normal(loc=5, scale=0.5, size=(100, 200)) for i in range(0, 3)]
    y_diff = np.random.laplace(loc=5, scale=0.5, size=(100, 200))
    seq_y_eval_diff = [np.random.normal(loc=5, scale=0.5, size=(100, 200)) for i in range(0, 3)]
    kernel = [(None, BasicRBFKernelFunction())]

    test_evaluator = StatsTestEvaluator(
        candidate_kernels=kernel,
        num_epochs=10,
        n_permutation_test=100)
    test_cases = test_evaluator.interface(code_approach='name-test',
                                          x_train=x_same,
                                          seq_x_eval=seq_x_eval,
                                          y_train_same=y_same,
                                          seq_y_eval_same=seq_y_eval_same,
                                          y_train_diff=y_diff,
                                          seq_y_eval_diff=seq_y_eval_diff,
                                          reg_lambda=0.001,
                                          reg_strategy='l1')
    # check if formatter passes
    formatter = TestResultGroupsFormatter(test_cases)
    formatter.format_result_summary_table()
    formatter.format_result_table()

    # check if
    p_temp_dir = mkdtemp()
    target_pickle = pathlib.Path(p_temp_dir, 'tests.pickle')
    target_torch = pathlib.Path(p_temp_dir, 'tests.torch')
    test_cases.save_test_results(target_pickle, 'pickle')
    test_cases.save_test_results(target_torch, 'torch')

    # check if mmd-estimator is usable again.
    x_test = seq_x_eval[0]
    y_test = seq_y_eval_diff[0]
    with target_pickle.open('rb') as f:
        tests_loaded_p = pickle.load(f)
        [m['mmd_estimator'].mmd_distance(x_test, y_test) for m in tests_loaded_p]
    with target_torch.open('rb') as f:
        tests_loaded_p = torch.load(f)
        [m['mmd_estimator'].mmd_distance(x_test, y_test) for m in tests_loaded_p]


if __name__ == '__main__':
    test_unit()
