from dataclasses import dataclass
import numpy as np
import torch

import typing
import tqdm

import math

from model_criticism_mmd.models.datasets import TwoSampleDataSet
from model_criticism_mmd import SelectionKernels, PermutationTest, MMD
from model_criticism_mmd.supports.selection_kernels import SelectedKernel
from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd.logger_unit import logger


@dataclass
class TestResult(object):
    codename_experiment: str
    kernel_parameter: float
    kernel: str
    is_same_distribution: bool
    ratio: float
    p_value: float

# todo
@dataclass
class TestResultGroups(object):
    pass
    # 実験/kernelごとにテーブルを作成する。True / Test result


class StatsTestEvaluator(object):
    def __init__(self,
                 candidate_kernels: typing.List[typing.Tuple[torch.Tensor, kernels_torch.BaseKernel]],
                 device_obj: torch.device,
                 num_epochs: int = 500,
                 n_permutation_test: int = 500,
                 initial_value_scales: typing.Optional[torch.Tensor] = None,
                 threshold_p_value: float = 0.05,
                 ratio_training: float = 0.8):
        self.candidate_kernels = candidate_kernels
        self.device_obj = device_obj
        self.num_epochs = num_epochs
        self.n_permutation_test = n_permutation_test
        self.initial_value_scales = initial_value_scales
        self.threshold_p_value = threshold_p_value
        self.ratio_training = ratio_training

    def function_separation(self, x: torch.Tensor, y: torch.Tensor) -> typing.Tuple[TwoSampleDataSet, TwoSampleDataSet]:
        ind_training = int((len(x) - 1) * self.ratio_training)
        x_train = x[:ind_training, :]
        x_val = x[ind_training:, :]
        y_train = y[:ind_training, :]
        y_val = y[ind_training:, :]

        dataset_train = TwoSampleDataSet(x_train, y_train, self.device_obj)
        dataset_val = TwoSampleDataSet(x_val, y_val, self.device_obj)

        return dataset_train, dataset_val

    def func_evaluation(self, p_value: float) -> bool:
        """

        Args:
            p_value: the given p-value

        Returns: True if same distribution, else different distribution.

        """
        return True if p_value > self.threshold_p_value else False

    def function_kernel_selection(self,
                                  ds_train: TwoSampleDataSet,
                                  ds_val: TwoSampleDataSet) -> typing.List[SelectedKernel]:
        """

        Args:
            ds_train:
            ds_val:

        Returns:

        """
        # kernel selection
        if self.initial_value_scales is None:
            init_scales = torch.tensor([1.0] * ds_train.get_dimension()[0])
        else:
            init_scales = self.initial_value_scales
            assert len(init_scales) == ds_train.get_dimension()
        # end if

        kernel_selector = SelectionKernels(num_epochs=self.num_epochs,
                                           dataset_validation=ds_val,
                                           device_obj=self.device_obj,
                                           is_training=True,
                                           dataset_training=ds_train,
                                           candidate_kernels=self.candidate_kernels)
        # todos
        selection_result = kernel_selector.run_selection(is_shuffle=False,
                                                         is_training_auto_stop=True,
                                                         num_workers=1)
        return selection_result

    def function_permutation_test(self, mmd_estimator: MMD, x: torch.Tensor, y: torch.Tensor
                                  ) -> typing.Tuple[PermutationTest, float]:
        """Runs permutation test."""
        dataset_for_permutation_test_data_sample = TwoSampleDataSet(x, y, self.device_obj)
        test_operator = PermutationTest(n_permutation_test=self.n_permutation_test,
                                                    mmd_estimator=mmd_estimator,
                                                    dataset=dataset_for_permutation_test_data_sample,
                                                    device_obj=self.device_obj)
        mmd_data_sample = test_operator.compute_statistic()
        p_value = test_operator.compute_p_value(mmd_data_sample)
        return test_operator, p_value

    def function_evaluation_all_kernels(self,
                                        x: torch.Tensor,
                                        y: torch.Tensor,
                                        mmd_estimators: typing.List[typing.Tuple[MMD, float]],
                                        code_approach: str,
                                        is_same_distribution: bool) -> typing.List[TestResult]:
        """Run permutation tests with the given all kernels.

        Args:
            x:
            y:
            mmd_estimators:
            code_approach:
            is_same_distribution:

        Returns:

        """
        results = []
        for estimator_obj, ratio in mmd_estimators:
            __test_operator, __p = self.function_permutation_test(estimator_obj, x, y)

            if isinstance(estimator_obj.kernel_function_obj, kernels_torch.MaternKernelFunction):
                kernel_param = estimator_obj.kernel_function_obj.nu
                name_kernel = f'{estimator_obj.kernel_function_obj.__class__.__name__}-nu={estimator_obj.kernel_function_obj.nu}'
            elif isinstance(estimator_obj.kernel_function_obj, kernels_torch.BasicRBFKernelFunction):
                kernel_param = estimator_obj.kernel_function_obj.log_sigma.detach().numpy()
                name_kernel = f'{estimator_obj.kernel_function_obj.__class__.__name__}-log-sigma={estimator_obj.kernel_function_obj.log_sigma}'
            else:
                kernel_param = math.nan
            # end if

            results.append(TestResult(codename_experiment=code_approach,
                                      kernel_parameter=kernel_param,
                                      kernel=f'{estimator_obj.kernel_function_obj.__class__.__name__}',
                                      is_same_distribution=is_same_distribution,
                                      ratio=ratio,
                                      p_value=__p))
        # end for
        return results

    def interface(self,
                  code_approach: str,
                  x: torch.Tensor,
                  y_same: torch.Tensor,
                  y_diff: torch.Tensor,
                  functions_no_optimization: typing.Optional[typing.List[kernels_torch.BaseKernel]] = None
                  ) -> typing.List[TestResult]:
        """Run permutation tests for cases where X=Y and X!=Y.

        Args:
            code_approach:
            x:
            y_same:
            y_diff:
            functions_no_optimization (optional): Kernel function without optimizations

        Returns: [TestResult]
        """
        # without normalization
        ds_train_same, ds_val_same = self.function_separation(x=x, y=y_same)
        ds_train_diff, ds_val_diff = self.function_separation(x=x, y=y_diff)

        kernels_same = self.function_kernel_selection(ds_train=ds_train_same, ds_val=ds_val_same)
        kernels_diff = self.function_kernel_selection(ds_train=ds_train_diff, ds_val=ds_val_diff)

        estimator_same = [
            (MMD.from_trained_parameters(k_obj.trained_mmd_parameter, self.device_obj), k_obj.test_power)
            for k_obj in kernels_same]
        estimator_diff = [
            (MMD.from_trained_parameters(k_obj.trained_mmd_parameter, self.device_obj), k_obj.test_power)
             for k_obj in kernels_diff]
        if functions_no_optimization is not None:
            estimator_same += [(k_obj, None) for k_obj in functions_no_optimization]
            estimator_diff += [(k_obj, None) for k_obj in functions_no_optimization]
        # end if

        tests_same = self.function_evaluation_all_kernels(x=x, y=y_same, mmd_estimators=estimator_same,
                                                          code_approach=code_approach,
                                                          is_same_distribution=True)
        tests_diff = self.function_evaluation_all_kernels(x=x, y=y_diff, mmd_estimators=estimator_same,
                                                          code_approach=code_approach,
                                                          is_same_distribution=False)

        return tests_same + tests_diff






eval_results = []


def sample_data_preparation():
    x_data_sample = np.zeros((N_DATA_SIZE, N_TIME_LENGHTH))
    y_data_sample = np.zeros((N_DATA_SIZE, N_TIME_LENGHTH))
    y_data_sample_laplase = np.zeros((N_DATA_SIZE, N_TIME_LENGHTH))

    x_data_sample[:, 0] = INITIAL_VALUE_AT_ONE
    y_data_sample[:, 0] = INITIAL_VALUE_AT_ONE
    y_data_sample_laplase[:, 0] = INITIAL_VALUE_AT_ONE

    for time_t in tqdm.tqdm(range(0, N_TIME_LENGHTH - 1)):
        noise_x = np.random.normal(NOISE_MU_X, NOISE_SIGMA_X, (N_DATA_SIZE,))
        noise_y = np.random.normal(NOISE_MU_Y, NOISE_SIGMA_Y, (N_DATA_SIZE,))
        noise_y_laplase = np.random.laplace(NOISE_MU_Y, NOISE_SIGMA_Y, (N_DATA_SIZE,))
        x_data_sample[:, time_t + 1] = x_data_sample[:, time_t].flatten() + noise_x
        y_data_sample[:, time_t + 1] = y_data_sample[:, time_t].flatten() + noise_y
        y_data_sample_laplase[:, time_t + 1] = y_data_sample_laplase[:, time_t].flatten() + noise_y_laplase
        # end if
    assert x_data_sample.shape == (N_DATA_SIZE, N_TIME_LENGHTH)
    assert y_data_sample.shape == (N_DATA_SIZE, N_TIME_LENGHTH)
    assert y_data_sample_laplase.shape == (N_DATA_SIZE, N_TIME_LENGHTH)
    assert np.array_equal(x_data_sample, y_data_sample) is False




