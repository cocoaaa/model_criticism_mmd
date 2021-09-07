from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction
from model_criticism_mmd import TestResultGroupsFormatter, StatsTestEvaluator
import numpy as np


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
                                          seq_y_eval_diff=seq_y_eval_diff)
    formatter = TestResultGroupsFormatter(test_cases)
    formatter.format_result_summary_table()
    formatter.format_result_table()


if __name__ == '__main__':
    test_unit()
