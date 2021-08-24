# def sample_data_preparation():
#     x_data_sample = np.zeros((N_DATA_SIZE, N_TIME_LENGHTH))
#     y_data_sample = np.zeros((N_DATA_SIZE, N_TIME_LENGHTH))
#     y_data_sample_laplase = np.zeros((N_DATA_SIZE, N_TIME_LENGHTH))
#
#     x_data_sample[:, 0] = INITIAL_VALUE_AT_ONE
#     y_data_sample[:, 0] = INITIAL_VALUE_AT_ONE
#     y_data_sample_laplase[:, 0] = INITIAL_VALUE_AT_ONE
#
#     for time_t in tqdm.tqdm(range(0, N_TIME_LENGHTH - 1)):
#         noise_x = np.random.normal(NOISE_MU_X, NOISE_SIGMA_X, (N_DATA_SIZE,))
#         noise_y = np.random.normal(NOISE_MU_Y, NOISE_SIGMA_Y, (N_DATA_SIZE,))
#         noise_y_laplase = np.random.laplace(NOISE_MU_Y, NOISE_SIGMA_Y, (N_DATA_SIZE,))
#         x_data_sample[:, time_t + 1] = x_data_sample[:, time_t].flatten() + noise_x
#         y_data_sample[:, time_t + 1] = y_data_sample[:, time_t].flatten() + noise_y
#         y_data_sample_laplase[:, time_t + 1] = y_data_sample_laplase[:, time_t].flatten() + noise_y_laplase
#         # end if
#     assert x_data_sample.shape == (N_DATA_SIZE, N_TIME_LENGHTH)
#     assert y_data_sample.shape == (N_DATA_SIZE, N_TIME_LENGHTH)
#     assert y_data_sample_laplase.shape == (N_DATA_SIZE, N_TIME_LENGHTH)
#     assert np.array_equal(x_data_sample, y_data_sample) is False

import numpy as np
import torch
from tabulate import tabulate

from model_criticism_mmd.supports import StatsTestEvaluator, TestResultGroupsFormatter
from model_criticism_mmd import kernels_torch


def test_evaluate_stats_tests():
    x = np.random.normal(0, 1.0, size=(500, 100))
    y_same = np.random.normal(0, 1.0, size=(500, 100))
    y_diff = np.random.laplace(0, 1.0, size=(500, 100))

    initial_scales = torch.tensor([1.0] * x.shape[-1])

    kernels_opt = [
        (initial_scales, kernels_torch.MaternKernelFunction(nu=2.5, lengthscale=-1.0))
    ]
    kernels_non_opt = [
        kernels_torch.MaternKernelFunction(2.5, lengthscale=-1.0)
    ]

    test_eval = StatsTestEvaluator(
        candidate_kernels=kernels_opt,
        kernels_no_optimization=kernels_non_opt,
        num_epochs=10
    )
    evals_1 = test_eval.interface(code_approach='test-1', x=x, y_same=y_same, y_diff=y_diff)
    evals_2 = test_eval.interface(code_approach='test-2', x=x, y_same=y_same, y_diff=y_diff)
    eval_formatter = TestResultGroupsFormatter(evals_1 + evals_2)

    result_text = eval_formatter.format_test_result_summary()
    print(result_text)
    result_table = eval_formatter.format_result_table()
    print(tabulate(result_table))
