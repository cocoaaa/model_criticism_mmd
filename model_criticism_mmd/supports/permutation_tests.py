#! - coding: utf-8 -*-
import torch

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.backend_torch import MMD
from model_criticism_mmd.models import MmdValues
from model_criticism_mmd.models.datasets import TwoSampleDataSet


class PermutationTest(object):
    def __init__(self,
                 mmd_estimator: MMD,
                 dataset: TwoSampleDataSet,
                 batch_size: int = 256,
                 device_obj: torch.device = torch.device('cpu'),
                 is_shuffle: bool = False):
        self.mmd_estimator = mmd_estimator
        self.dataset = dataset
        self.batch_size = batch_size
        self.device_obj = device_obj
        self.is_shuffle = is_shuffle

    def get_kernel_matrix(self) -> float:
        """

        The implementation follows SGMatrix<float32_t> QuadraticTimeMMD::Self::get_kernel_matrix().
        Ref: https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/QuadraticTimeMMD.cpp#L169


        Returns:

        """
        pass


    def compute_p_value(self, statistics):
        """
        float64_t QuadraticTimeMMD::compute_p_value(float64_t statistic)
        {
            require(get_kernel(), "Kernel is not set!");
            float64_t result=0;
            switch (get_null_approximation_method())
            {
                case NAM_MMD2_GAMMA:
                {
                    SGVector<float64_t> params=self->gamma_fit_null();
                    result=Statistics::gamma_cdf(statistic, params[0], params[1]);
                    break;
                }
                default:
                    result=HypothesisTest::compute_p_value(statistic);
                break;
            }
            return result;
        }
        """

    def compute_threshold(self, alpha: float = 0.05):
        """
        float64_t
        QuadraticTimeMMD::compute_threshold(float64_t
        alpha)
        {
            require(get_kernel(), "Kernel is not set!");
        float64_t
        result = 0;
        switch(get_null_approximation_method())
        {
            case
        NAM_MMD2_GAMMA:
        {
            SGVector < float64_t > params = self->gamma_fit_null();
        result = Statistics::gamma_inverse_cdf(alpha, params[0], params[1]);
        break;
        }
        default:
        result = HypothesisTest::compute_threshold(alpha);
        break;

    }
    return result;
    }
    """
    def statistic_job(self):
        """
        # note
        # what is statistic_job??
        # a class of computeMMD. https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/internals/mmd/ComputeMMD.h
        # ComputeMMD statistic_job; https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/QuadraticTimeMMD.cpp#L97
        # to call the method, an initialization is mandatory: https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/QuadraticTimeMMD.cpp#L116-L126

        The actual operation is here
        https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/internals/mmd/ComputeMMD.h#L85-L121
        """

    def __compute_mmd(self, num_workers: int = 1) -> torch.Tensor:
        if len(self.dataset) < self.batch_size:
            x, y = self.dataset.get_all_item()
            return self.mmd_estimator.mmd_distance(x, y).mmd
        else:
            total_mmd2_val = 0
            n_batches = 0
            if self.device_obj == torch.device('cpu'):
                data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                          shuffle=self.is_shuffle,
                                                          num_workers=num_workers)
            else:
                data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                          shuffle=self.is_shuffle)
            # end if
            for xbatch, ybatch in data_loader:
                mmd_values = self.mmd_estimator.mmd_distance(xbatch, ybatch)
                total_mmd2_val += mmd_values.mmd
                n_batches += 1
            # end for
            avg_mmd2 = torch.div(total_mmd2_val, n_batches)
            return avg_mmd2

    def compute_statistic(self) -> torch.Tensor:
        """computing MMD value for whole dataset.
        Note: if N(data) > batch_size, statistics is avg(mmd) over batch_size"""
        statistics = self.__compute_mmd()
        return statistics
