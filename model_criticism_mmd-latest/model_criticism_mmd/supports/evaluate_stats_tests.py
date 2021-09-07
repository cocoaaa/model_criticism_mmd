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
    distributions_permutation_test: torch.Tensor
    statistics_whole: float


@dataclass
class TestResultGroupsFormatter(object):
    test_result: typing.List[TestResult]

    @staticmethod
    def __function_test_result_type(record: typing.Union[pandas.DataFrame, typing.Dict]) -> str:
        if record['is_same_distribution_truth'] and record['is_same_distribution_test']:
            return 'pass'
        elif record['is_same_distribution_truth'] is False and record['is_same_distribution_test'] is False:
            return 'pass'
        elif record['is_same_distribution_truth'] is True and record['is_same_distribution_test'] is False:
            return 'error_type-1'
        elif record['is_same_distribution_truth'] is False and record['is_same_distribution_test'] is True:
            return 'error_type-2'
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
        df_res['test_result'] = df_res.apply(self.__function_test_result_type, axis=1)
        df_output = df_res.reindex(columns=['codename_experiment', 'kernel', 'kernel_parameter', 'is_optimized',
                                            'test_result', 'p_value', 'is_same_distribution_truth',
                                            'is_same_distribution_test', 'ratio'])
        return df_output

    def format_result_summary_table(self) -> pandas.DataFrame:
        """Return a table that includes result of X=Y and X!=Y.
        Aggregated by "code-name", "kernel-name" and "is_optimization"

        Returns: DataFrame
        """
        summary_record = []
        for t_key, records in itertools.groupby(
                sorted(self.test_result, key=lambda r: (r.codename_experiment, r.kernel, r.is_optimized)),
                key=lambda rr: (rr.codename_experiment, rr.kernel, rr.is_optimized)):
            seq_records = list(records)
            new_record = {
                'test-key': f'{t_key[0]}-{t_key[1]}-{t_key[2]}',
                'X=Y_total': 0,
                'X=Y_pass': 0,
                'X=Y_error-1': 0,
                'X=Y_error-2': 0,
                'X!=Y_total': 0,
                'X!=Y_pass': 0,
                'X!=Y_error-1': 0,
                'X!=Y_error-2': 0,
                'kernel': seq_records[0].kernel,
                'length_scale': seq_records[0].kernel_parameter,
                'is_optimization': seq_records[0].is_optimized
            }
            for r in seq_records:
                class_test_result = self.__function_test_result_type(self.asdict(r))
                if r.is_same_distribution_truth:
                    new_record['X=Y_total'] += 1
                    if class_test_result == 'pass':
                        new_record['X=Y_pass'] += 1
                    elif class_test_result == 'error_type-1':
                        new_record['X=Y_error-1'] += 1
                    elif class_test_result == 'error_type-2':
                        new_record['X=Y_error-2'] += 1
                    else:
                        raise NotImplementedError()
                    # end if
                else:
                    new_record['X!=Y_total'] += 1
                    if class_test_result == 'pass':
                        new_record['X!=Y_pass'] += 1
                    elif class_test_result == 'error_type-1':
                        new_record['X!=Y_error-1'] += 1
                    elif class_test_result == 'error_type-2':
                        new_record['X!=Y_error-2'] += 1
                    else:
                        raise NotImplementedError()
                    # end if
                # end if
            # end for
            summary_record.append(new_record)
        # end for
        df_res = pandas.DataFrame(summary_record)
        return df_res


class StatsTestEvaluator(object):
    def __init__(self,
                 candidate_kernels: typing.List[typing.Tuple[typing.Optional[torch.Tensor], kernels_torch.BaseKernel]],
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
                                  ) -> typing.Tuple[PermutationTest, float, float]:
        """Runs permutation test."""
        dataset_for_permutation_test_data_sample = TwoSampleDataSet(x, y, self.device_obj)
        test_operator = PermutationTest(n_permutation_test=self.n_permutation_test,
                                                    mmd_estimator=mmd_estimator,
                                                    dataset=dataset_for_permutation_test_data_sample,
                                                    device_obj=self.device_obj)
        mmd_data_sample = test_operator.compute_statistic()
        p_value = test_operator.compute_p_value(mmd_data_sample)
        if isinstance(p_value, torch.Tensor):
            p_value = p_value.detach().cpu().numpy()
        # end
        if isinstance(mmd_data_sample, torch.Tensor):
            mmd_data_sample = mmd_data_sample.detach().cpu().numpy()
        return test_operator, p_value, mmd_data_sample

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
            __test_operator, __p, __mmd_whole = self.function_permutation_test(estimator_obj, x, y)
            distributions_test = __test_operator.stats_permutation_test
            if isinstance(distributions_test, torch.Tensor):
                distributions_test = distributions_test.detach().cpu().numpy()
            # end
            if isinstance(estimator_obj.kernel_function_obj, kernels_torch.MaternKernelFunction):
                kernel_param = estimator_obj.kernel_function_obj.gpy_kernel.lengthscale.detach().cpu().numpy()
                name_kernel = f'{estimator_obj.kernel_function_obj.__class__.__name__}-nu={estimator_obj.kernel_function_obj.nu}'
            elif isinstance(estimator_obj.kernel_function_obj, kernels_torch.BasicRBFKernelFunction):
                kernel_param = estimator_obj.kernel_function_obj.log_sigma.detach().cpu().numpy()
                name_kernel = f'{estimator_obj.kernel_function_obj.__class__.__name__}'
            else:
                kernel_param = math.nan
                name_kernel = 'undefined'
            # end if
            is_same_test = self.func_evaluation(__p)
            if isinstance(estimator_obj.scales, torch.Tensor):
                scales = estimator_obj.scales.detach().cpu().numpy()
            else:
                scales = estimator_obj.scales
            # end

            results.append(TestResult(codename_experiment=code_approach,
                                      kernel_parameter=kernel_param,
                                      kernel=name_kernel,
                                      is_same_distribution_truth=is_same_distribution,
                                      is_same_distribution_test=is_same_test,
                                      is_optimized=False if ratio is None else True,
                                      ratio=ratio,
                                      p_value=__p,
                                      scales=scales,
                                      distributions_permutation_test=distributions_test,
                                      statistics_whole=__mmd_whole))
        # end for
        return results

    def interface(self,
                  code_approach: str,
                  x_train: typing.Union[torch.Tensor, np.ndarray],
                  seq_x_eval: typing.List[typing.Union[torch.Tensor, np.ndarray]],
                  y_train_same: typing.Optional[typing.Union[torch.Tensor, np.ndarray]] = None,
                  y_train_diff: typing.Optional[typing.Union[torch.Tensor, np.ndarray]] = None,
                  seq_y_eval_same: typing.Optional[typing.List[typing.Union[torch.Tensor, np.ndarray]]] = None,
                  seq_y_eval_diff: typing.Optional[typing.List[typing.Union[torch.Tensor, np.ndarray]]] = None
                  ) -> typing.List[TestResult]:
        """Run permutation tests for cases where X=Y and X!=Y.

        Args:
            code_approach:
            x_train: X data for training.
            seq_x_eval: List of X data for evaluation.
            y_train_same: Y data for training. Y is from the same distribution.
            y_train_diff: Y data for training. Y is from the different distribution.
            seq_y_eval_same: List of Y-same data for evaluation.
            seq_y_eval_diff: List of Y-diff for evaluation.

        Returns: [TestResult]
        """
        test_result = []

        if y_train_same is None and y_train_diff is None:
            raise Exception('Either of y_train_same or y_train_diff should be given.')

        if y_train_same is not None:
            assert seq_y_eval_same is not None
            assert len(seq_x_eval) == len(seq_y_eval_same), f'different length of eval-data. ' \
                                                            f'len(seq_x_eval): {len(seq_x_eval)} ' \
                                                            f'len(seq_y_eval_same): {len(seq_y_eval_same)}'
            ds_train_same, ds_val_same = self.function_separation(x=x_train, y=y_train_same)
            kernels_same = self.function_kernel_selection(ds_train=ds_train_same, ds_val=ds_val_same)
            estimator_same = [
                (MMD.from_trained_parameters(k_obj.trained_mmd_parameter, self.device_obj), k_obj.test_power)
                for k_obj in kernels_same]
            if self.kernels_no_optimization is not None:
                estimator_same += [(MMD(k_obj), None) for k_obj in self.kernels_no_optimization]
            # end if
            i = 0
            for x_eval, y_eval_same in zip(seq_x_eval, seq_y_eval_same):
                i += 1
                logger.info(f'Running X=Y {i} of {len(seq_x_eval)}')
                tests_same = self.function_evaluation_all_kernels(x=x_eval, y=y_eval_same,
                                                                  mmd_estimators=estimator_same,
                                                                  code_approach=code_approach,
                                                                  is_same_distribution=True)
                test_result += tests_same
            # end for
        # end if
        if y_train_diff is not None:
            assert seq_y_eval_diff is not None
            assert len(seq_x_eval) == len(seq_y_eval_diff), f'different length of eval-data. ' \
                                                            f'len(seq_x_eval): {len(seq_x_eval)} ' \
                                                            f'len(seq_y_eval_diff): {len(seq_y_eval_diff)}'
            ds_train_diff, ds_val_diff = self.function_separation(x=x_train, y=y_train_diff)
            kernels_diff = self.function_kernel_selection(ds_train=ds_train_diff, ds_val=ds_val_diff)
            estimator_diff = [
                (MMD.from_trained_parameters(k_obj.trained_mmd_parameter, self.device_obj), k_obj.test_power)
                for k_obj in kernels_diff]
            if self.kernels_no_optimization is not None:
                estimator_diff += [(MMD(k_obj), None) for k_obj in self.kernels_no_optimization]
            # end if
            i = 0
            for x_eval, y_eval_diff in zip(seq_x_eval, seq_y_eval_diff):
                i += 1
                logger.info(f'Running X!=Y {i} of {len(seq_x_eval)}')
                tests_diff = self.function_evaluation_all_kernels(x=x_eval, y=y_eval_diff,
                                                                  mmd_estimators=estimator_diff,
                                                                  code_approach=code_approach,
                                                                  is_same_distribution=False)
                test_result += tests_diff
            # end for
        # end if

        return test_result

