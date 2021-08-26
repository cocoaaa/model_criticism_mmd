import pandas
from dataclasses import dataclass
import numpy as np
import torch

import typing
import itertools

import math

from model_criticism_mmd.models.datasets import TwoSampleDataSet
from model_criticism_mmd.supports.selection_kernels import SelectedKernel, SelectionKernels, MMD
from model_criticism_mmd.supports.permutation_tests import PermutationTest
from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd.models import DEFAULT_DEVICE
from model_criticism_mmd.logger_unit import logger

from tabulate import tabulate

@dataclass
class TestResult(object):
    codename_experiment: str
    kernel_parameter: float
    kernel: str
    is_optimized: bool
    is_same_distribution_truth: bool
    is_same_distribution_test: bool
    ratio: float
    p_value: float
    scales: torch.Tensor


@dataclass
class TestResultGroupsFormatter(object):
    test_result: typing.List[TestResult]

    def format_test_result_summary(self) -> str:
        result_text = ''

        for k, g_obj in itertools.groupby(sorted(self.test_result, key=lambda o: (o.codename_experiment, o.kernel)),
                                           key=lambda o: (o.codename_experiment, o.kernel)):
            for test_result in g_obj:
                title_table = f'exp-code={test_result.codename_experiment}, Kernel={test_result.kernel} ' \
                              f'with length_scale={test_result.kernel_parameter} optimization={test_result.is_optimized}\n' \
                              f'p-value={test_result.p_value}'

                result_table = [[False, False], [False, False]]
                if test_result.is_same_distribution_test and test_result.is_same_distribution_truth:
                    result_table[0][0] = True
                elif test_result.is_same_distribution_truth and test_result.is_same_distribution_test is False:
                    result_table[0][1] = True
                elif test_result.is_same_distribution_truth is False and test_result.is_same_distribution_test is True:
                    result_table[1][0] = True
                elif test_result.is_same_distribution_truth is False and test_result.is_same_distribution_test is False:
                    result_table[1][1] = True
                else:
                    raise NotImplementedError('undefined')
                # end if

                df_out = pandas.DataFrame(result_table, index=[True, False], columns=[True, False])
                df_out.index.name = 'Truth / Test'

                result_text += title_table
                result_text += '\n'
                result_text += tabulate(df_out, headers='keys', tablefmt='psql')
                result_text += '\n\n'
            # end for
        # end for
        return result_text

    @staticmethod
    def function_test_result_type(record: typing.Union[pandas.DataFrame, typing.Dict]) -> str:
        if record['is_same_distribution_truth'] and record['is_same_distribution_test']:
            return 'pass'
        elif record['is_same_distribution_truth'] is False and record['is_same_distribution_test'] is False:
            return 'pass'
        elif record['is_same_distribution_truth'] is True and record['is_same_distribution_test'] is False:
            return 'error type-1'
        elif record['is_same_distribution_truth'] is False and record['is_same_distribution_test'] is True:
            return 'error type-2'
        else:
            raise NotImplementedError('undefined')

    @staticmethod
    def asdict(o: object) -> typing.Dict[str, typing.Any]:
        return {k: v for k, v in o.__dict__.items() if not k == 'scales'}

    def format_result_table(self) -> pandas.DataFrame:
        # A method to output X=Y / X!= Y all passed.
        # classification test pass / type-1 / type-2
        records = [self.asdict(r) for r in self.test_result]

        df_res = pandas.DataFrame(records)
        df_res['test_result'] = df_res.apply(self.function_test_result_type, axis=1)
        df_output = df_res.reindex(columns=['codename_experiment', 'kernel', 'kernel_parameter', 'is_optimized',
                                            'test_result', 'p_value', 'is_same_distribution_truth',
                                            'is_same_distribution_test', 'ratio'])
        return df_output

    def format_result_summary_table(self) -> pandas.DataFrame:
        """Return a table that includes result of X=Y and X!=Y.
        Aggregated by "code-name", "kernel-name" and "is_optimization"

        Returns: DataFrame
        """
        records = [self.asdict(r) for r in self.test_result]
        summary_record = []
        for t_key, records in itertools.groupby(
                sorted(self.test_result, key=lambda r: (r.codename_experiment, r.kernel, r.is_optimized)),
                key=lambda r: (r.codename_experiment, r.kernel, r.is_optimized)):
            new_record = {
                'test-key': f'{t_key[0]}-{t_key[1]}-{t_key[2]}',
                'X=Y': None,
                'X!=Y': None
            }
            for r in records:
                if r.is_same_distribution_truth:
                    # X=Y
                    if r.is_same_distribution_test is True:
                        new_record['X=Y'] = 'pass'
                    else:
                        new_record['X=Y'] = 'error type-1'
                else:
                    # X!=Y
                    if r.is_same_distribution_test is False:
                        new_record['X!=Y'] = 'pass'
                    else:
                        new_record['X!=Y'] = 'error type-2'
                # end if
            # end for
            summary_record.append(new_record)
        # end for
        df_res = pandas.DataFrame(summary_record)
        return df_res


class StatsTestEvaluator(object):
    def __init__(self,
                 candidate_kernels: typing.List[typing.Tuple[torch.Tensor, kernels_torch.BaseKernel]],
                 device_obj: torch.device = DEFAULT_DEVICE,
                 num_epochs: int = 500,
                 n_permutation_test: int = 500,
                 initial_value_scales: typing.Optional[torch.Tensor] = None,
                 threshold_p_value: float = 0.05,
                 ratio_training: float = 0.8,
                 kernels_no_optimization: typing.Optional[typing.List[kernels_torch.BaseKernel]] = None):
        self.candidate_kernels = candidate_kernels
        self.device_obj = device_obj
        self.num_epochs = num_epochs
        self.n_permutation_test = n_permutation_test
        self.initial_value_scales = initial_value_scales
        self.threshold_p_value = threshold_p_value
        self.ratio_training = ratio_training
        self.kernels_no_optimization = kernels_no_optimization

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
                kernel_param = estimator_obj.kernel_function_obj.gpy_kernel.lengthscale
                name_kernel = f'{estimator_obj.kernel_function_obj.__class__.__name__}-nu={estimator_obj.kernel_function_obj.nu}'
            elif isinstance(estimator_obj.kernel_function_obj, kernels_torch.BasicRBFKernelFunction):
                kernel_param = estimator_obj.kernel_function_obj.log_sigma.detach().numpy()
                name_kernel = f'{estimator_obj.kernel_function_obj.__class__.__name__}'
            else:
                kernel_param = math.nan
                name_kernel = 'undefined'
            # end if
            is_same_test = self.func_evaluation(__p)
            results.append(TestResult(codename_experiment=code_approach,
                                      kernel_parameter=kernel_param,
                                      kernel=name_kernel,
                                      is_same_distribution_truth=is_same_distribution,
                                      is_same_distribution_test=is_same_test,
                                      is_optimized=False if ratio is None else True,
                                      ratio=ratio,
                                      p_value=__p,
                                      scales=estimator_obj.scales))
        # end for
        return results

    def interface(self,
                  code_approach: str,
                  x_train: typing.Union[torch.Tensor, np.ndarray],
                  x_eval: typing.Union[torch.Tensor, np.ndarray],
                  y_train_same: typing.Optional[typing.Union[torch.Tensor, np.ndarray]] = None,
                  y_train_diff: typing.Optional[typing.Union[torch.Tensor, np.ndarray]] = None,
                  y_eval_same: typing.Optional[typing.Union[torch.Tensor, np.ndarray]] = None,
                  y_eval_diff: typing.Optional[typing.Union[torch.Tensor, np.ndarray]] = None
                  ) -> typing.List[TestResult]:
        """Run permutation tests for cases where X=Y and X!=Y.

        Args:
            code_approach:
            x_train: X data for training.
            x_eval: X data for evaluation.
            y_train_same: Y data for training. Y is from the same distribution.
            y_train_diff: Y data for training. Y is from the different distribution.
            y_eval_same: omit
            y_eval_diff: omit

        Returns: [TestResult]
        """
        test_result = []

        if y_train_same is None and y_train_diff is None:
            raise Exception('Either of y_train_same or y_train_diff should be given.')

        if y_train_same is not None:
            assert y_eval_same is not None
            ds_train_same, ds_val_same = self.function_separation(x=x_train, y=y_train_same)
            kernels_same = self.function_kernel_selection(ds_train=ds_train_same, ds_val=ds_val_same)
            estimator_same = [
                (MMD.from_trained_parameters(k_obj.trained_mmd_parameter, self.device_obj), k_obj.test_power)
                for k_obj in kernels_same]
            if self.kernels_no_optimization is not None:
                estimator_same += [(MMD(k_obj), None) for k_obj in self.kernels_no_optimization]
            # end if
            tests_same = self.function_evaluation_all_kernels(x=x_eval, y=y_eval_same,
                                                              mmd_estimators=estimator_same,
                                                              code_approach=code_approach,
                                                              is_same_distribution=True)
            test_result += tests_same
        # end if
        if y_train_diff is not None:
            assert y_eval_diff is not None
            ds_train_diff, ds_val_diff = self.function_separation(x=x_train, y=y_train_diff)
            kernels_diff = self.function_kernel_selection(ds_train=ds_train_diff, ds_val=ds_val_diff)
            estimator_diff = [
                (MMD.from_trained_parameters(k_obj.trained_mmd_parameter, self.device_obj), k_obj.test_power)
                for k_obj in kernels_diff]
            if self.kernels_no_optimization is not None:
                estimator_diff += [(MMD(k_obj), None) for k_obj in self.kernels_no_optimization]
            # end if
            tests_diff = self.function_evaluation_all_kernels(x=x_eval, y=y_eval_diff,
                                                              mmd_estimators=estimator_diff,
                                                              code_approach=code_approach,
                                                              is_same_distribution=False)
            test_result += tests_diff
        # end if

        return test_result

