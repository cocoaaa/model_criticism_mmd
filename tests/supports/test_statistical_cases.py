import numpy as np
import torch

from model_criticism_mmd.supports import StatsTestEvaluator, TestResultGroupsFormatter
from model_criticism_mmd import kernels_torch
from model_criticism_mmd.models.static import DEFAULT_DEVICE


def test_evaluate_stats_tests():
    x = np.random.normal(0, 1.0, size=(500, 100))
    y_same = np.random.normal(0, 1.0, size=(500, 100))
    y_diff = np.random.laplace(0, 1.0, size=(500, 100))

    # evals
    x_eval = [np.random.normal(0, 1.0, size=(500, 100)) for i in range(0, 3)]
    y_same_eval = [np.random.normal(0, 1.0, size=(500, 100)) for i in range(0, 3)]
    y_diff_eval = [np.random.laplace(0, 1.0, size=(500, 100)) for i in range(0, 3)]

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
        num_epochs=5,
        n_permutation_test=100,
        device_obj=DEFAULT_DEVICE
    )
    evals_1 = test_eval.interface(code_approach='test-1', x_train=x, seq_x_eval=x_eval,
                                  y_train_same=y_same, seq_y_eval_same=y_same_eval)
    evals_2 = test_eval.interface(code_approach='test-1', x_train=x, seq_x_eval=x_eval,
                                  y_train_diff=y_diff, seq_y_eval_diff=y_diff_eval)
    eval_formatter = TestResultGroupsFormatter(evals_1 + evals_2)

    result_table = eval_formatter.format_result_table()
    result_summary_table = eval_formatter.format_result_summary_table()
    assert len(result_summary_table) == 2


if __name__ == '__main__':
    test_evaluate_stats_tests()
