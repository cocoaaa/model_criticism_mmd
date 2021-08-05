#! - coding: utf-8 -*-
import torch
import math
import typing
import numpy as np
from tqdm import tqdm

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.backend_torch import MMD
from model_criticism_mmd.models import MmdValues
from model_criticism_mmd.models.datasets import TwoSampleDataSet


class PermutationTest(object):
    def __init__(self,
                 mmd_estimator: MMD,
                 dataset: TwoSampleDataSet,
                 n_permutation_test: int = 1000,
                 batch_size: int = 256,
                 device_obj: torch.device = torch.device('cpu'),
                 is_shuffle: bool = False,
                 is_mmd2_gamma: bool = False,
                 is_normalize: bool = False):
        """A class to run Permutation test.

        Examples:
            >>> mmd_estimator = MMD(kernel_function_obj=kernel_function, device_obj=device_obj, scales=init_scale)
            >>> dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
            >>> dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)
            >>> permutation_tester = PermutationTest(mmd_estimator=mmd_estimator, dataset=dataset_train)
            >>> statistics = permutation_tester.compute_statistic()
            >>> threshold = permutation_tester.compute_threshold(alpha=0.05)
            >>> p_value = permutation_tester.compute_p_value(statistics)

        Args:
            mmd_estimator: MMD object with a specified kernel.
            dataset: a dataset.
            batch_size: a batch size to compute MMD value for the whole dataset.
            is_shuffle: a option in computing MMD value for the whole dataset.
            device_obj: cuda or cpu
            is_mmd2_gamma: for a very fast, but not consistent test based on moment matching of a Gamma distribution, as described in
             Arthur Gretton et. al. "Optimal kernel choice for large-scale two-sample tests". NIPS 2012: 1214-1222.
            is_normalize:
        """

        self.mmd_estimator = mmd_estimator
        self.dataset = dataset
        self.batch_size = batch_size
        self.device_obj = device_obj
        self.is_shuffle = is_shuffle
        self.is_mmd2_gamma = is_mmd2_gamma
        self.n_permutation_test = n_permutation_test
        self.stats_permutation_test: typing.Optional[np.ndarray] = None
        self.is_normalize = is_normalize

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

    def sample_null(self,
                    num_permutations: int = 250,
                    is_progress: bool = True) -> np.ndarray:
        """Compute statistics with sub-sampled data. The process is normally called "Permutation".

        Args:
            num_permutations: #time to run permutation test.
            is_progress: boolen, if True; then show a progress bar, False not.

        Returns: array object which contains values by permutations.
        """
        all_samples = self.dataset.get_all_item()
        x_all = all_samples[0]
        y_all = all_samples[1]
        z = torch.cat([x_all, y_all], 0)  # SOLUTION
        n_x = x_all.shape[0]
        n_y = y_all.shape[0]
        stats = []

        range_ = range(num_permutations)
        if is_progress:
            range_ = tqdm(range_)

        for i in range_:
            np.random.shuffle(z.numpy())  # SOLUTION
            mmd_values = self.mmd_estimator.mmd_distance(z[:n_x], z[n_x:])
            if self.is_normalize:
                stats.append(self.normalize_statistic(mmd_values.mmd).detach().cpu().numpy())
            else:
                stats.append(mmd_values.mmd.detach().cpu().numpy())
        # end for
        return np.array(stats)

    def compute_threshold(self, alpha: float = 0.05) -> float:
        """Compute threshold against the given alpha value.

        Args:
            alpha:

        Returns: A value where the threshold exists.
        """
        if self.is_mmd2_gamma:
            raise NotImplementedError('Not implemented yet')
        else:
            if self.stats_permutation_test is None:
                self.stats_permutation_test = self.sample_null(num_permutations=self.n_permutation_test)
            # end if
            values_sorted = sorted(self.stats_permutation_test)
            index_alpha: int = math.floor(len(values_sorted) * (1 - alpha))
            return values_sorted[index_alpha]

    def normalize_statistic(self, statistic: torch.Tensor) -> torch.Tensor:
        n_x = self.dataset.length_x
        n_y = self.dataset.length_y
        return n_x * n_y * statistic / (n_x + n_y)

    def compute_statistic(self) -> np.ndarray:
        """computing MMD value for whole dataset.
        Note: if N(data) > batch_size, statistics is avg(mmd) over batch_size"""
        statistics = self.__compute_mmd()
        if self.is_normalize:
            normalized_statistic = self.normalize_statistic(statistics)
            return normalized_statistic.detach().cpu().numpy()
        else:
            return statistics.detach().numpy()

    def compute_p_value(self, statistic: np.ndarray) -> float:
        """Compute p-value based on the permutation tests.

        Args:
            statistic: A statistic value with the whole dataset.

        Returns: p-value.
        """
        if self.stats_permutation_test is None:
            self.stats_permutation_test = self.sample_null(num_permutations=self.n_permutation_test)
        # end if
        values_sorted = np.sort(self.stats_permutation_test)
        i = self.find_position_to_insert(values_sorted, statistic)
        return 1.0 - i / len(self.stats_permutation_test)

    @staticmethod
    def find_position_to_insert(values_sorted: np.ndarray, statistic: np.ndarray) -> int:
        i = 0
        for i, value_i in enumerate(values_sorted):
            if value_i > statistic:
                return i
            # end if
        # end if
        return i

'''
class PermutationTestEfficient(object):
    def __init__(self,
                 mmd_estimator: MMD,
                 dataset: TwoSampleDataSet,
                 batch_size: int = 256,
                 device_obj: torch.device = torch.device('cpu'),
                 is_shuffle: bool = False,
                 is_mmd2_gamma: bool = False):
        """

        Args:
            mmd_estimator:
            dataset:
            batch_size:
            device_obj:
            is_shuffle:
            is_mmd2_gamma: for a very fast, but not consistent test based on moment matching of a Gamma distribution, as described in
             Arthur Gretton et. al. "Optimal kernel choice for large-scale two-sample tests". NIPS 2012: 1214-1222.
        """

        self.mmd_estimator = mmd_estimator
        self.dataset = dataset
        self.batch_size = batch_size
        self.device_obj = device_obj
        self.is_shuffle = is_shuffle
        self.is_mmd2_gamma = is_mmd2_gamma

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

    def precompute_permutation_inds(self):
        # corresponding between original kernel matricies and k-matricies for permutation tests.
        """
        ASSERT(m_num_null_samples > 0);
        # 対応関係Matrixの作成
        allocate_permutation_inds();
        for (auto n=0; n < m_num_null_samples; ++n)
            {
                # m_permuted_inds is a vector object.
                # note: initialize a vector from 0, the vector has the same size size = (n_x + n_y)
                std:: iota(m_permuted_inds.data(), m_permuted_inds.data() + m_permuted_inds.size(), 0);
            # shuffle the vector m_permuted_inds
            CMath::permute(m_permuted_inds);
            if (m_save_inds)
            {
            auto offset=n * m_permuted_inds.size();
            std::
                copy(m_permuted_inds.data(), m_permuted_inds.data() + m_permuted_inds.size(), & m_all_inds.matrix[
                    offset]);
            }
            for (index_t i=0; i < m_permuted_inds.size(); ++i)
            # note: m_inverted_permuted_inds is a matrix for corresponding.
            row=shuffled index of original data / col=an index of vector with that we compute MMD in permutation test / value=pointer to original data(where size=n_x+n_y)
            # note: m_inverted_permuted_inds is a matrix that (n-sample of permuted sample, length-of-vector for one permutation-test)


            m_inverted_permuted_inds(m_permuted_inds[i], n)=i;
            }
            """

    def sample_null(self):
        """Method that returns a number of null-samples, based on the null approximation method that was set using set_null_approximation_method().
        Default is permutation.

        Returns:
        """
        # todo implementation of NAM_MMD2_SPECTRUM if possible. Right now, only permutation test
        # ref: https://www.shogun-toolbox.org/api/latest/QuadraticTimeMMD_8cpp_source.html#l00570
        kernel_matrix = get_kernel_matrix()
        # todo init permutation job
        # note: n of X
        permutation_job.m_n_x = owner.get_num_samples_p()
        # note: n of Y
        permutation_job.m_n_y = owner.get_num_samples_q()
        # todo: what??
        permutation_job.m_stype = owner.get_statistic_type()
        # todo: how to decide?
        # default is 250. static constexpr index_t DEFAULT_NUM_NULL_SAMPLES = 250;
        permutation_job.m_num_null_samples = owner.get_num_null_samples()

        self.precompute_permutation_inds()

        # Note: permutation test follows implementation of PermutationMMD.h in Shogun.h
        # implementation of permutation-test is in
        # https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/internals/mmd/PermutationMMD.h#L59-L88
        # todo permutation manager is in another class?

        n_x = num_x
        n_y = num_y
        size = n_x + n_y
        for j in range(0, size):
            for i in range(j, size):
                pass

        """
        # This method exists in PermutationMMD class which inherits ComputeMMD class.
        # note: 以下の操作はおそらくKernel Matrixを固定している。Kernel Matrixを固定し、参照indexを変更している。その後、MMDを計算。
        # note, procedure.
        # 1. 
        # 3種類の場合わけが発生する。どのmatrixを見るべきなのか、決定する必要あり。x,x と y,y と x,y
        
        
        	SGVector<float32_t> operator()(const Kernel& kernel, PRNG& prng)
        {
            ASSERT(m_n_x>0 && m_n_y>0);
            ASSERT(m_num_null_samples>0);
            # permutation用の対応関係をもつmatrixを作成する。
            # https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/internals/mmd/PermutationMMD.h#L203-L219
            precompute_permutation_inds(prng);
    
            const index_t size=m_n_x+m_n_y;
            SGVector<float32_t> null_samples(m_num_null_samples);
    #pragma omp parallel for
            
            # m_num_null_samples が繰り返し回数を定義
            for (auto n=0; n<m_num_null_samples; ++n)
            {
                # struct in https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/internals/mmd/ComputeMMD.h#L56-L60
                terms_t terms;
                # size: XとYのサンプル数合計
                for (auto j=0; j<size; ++j)
                {
                    # 新しいk matrixにおけるcol index
                    auto inverted_col=m_inverted_permuted_inds(j, n);
                    # size: XとYのサンプル数合計
                    for (auto i=j; i<size; ++i)
                    {
                        # 
                        auto inverted_row=m_inverted_permuted_inds(i, n);
    
                        if (inverted_row>=inverted_col)
                            # note: 値を割り当てるk-matrixを選択する。k_xx, k_yy, k_xy 
                            # https://github.com/shogun-toolbox/shogun/blob/9b8d856971af5a295dd6ad70623ae45647a6334c/src/shogun/statistical_testing/internals/mmd/ComputeMMD.h#L150-L184
                            add_term_lower(terms, kernel(i, j), inverted_row, inverted_col);
                        else
                            add_term_lower(terms, kernel(i, j), inverted_col, inverted_row);
                    }
                }
                # note: compute method is in ComputeMMD class
                # note: run computation of MMD. terms is an arguments which represent Kernel objects.
                null_samples[n]=compute(terms);
                SG_DEBUG("null_samples[{}] = {}!", n, null_samples[n]);
            }
            return null_samples;
        }
        """
        result = permutation_job(kernel_matrix)
        return result

    def compute_threshold(self, alpha: float = 0.05):
        if self.is_mmd2_gamma:
            params = gamma_fit_null()
            threshold = gamma_inverse_cdf(alpha, params[0], params[1])
        else:
            values = sample_null()
            values_sorted = sorted(values)
            index_alpha: int = math.floor(len(values_sorted) * (1 - alpha))
            return values_sorted[index_alpha]

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

    def normalize_statistic(self, statistic: torch.Tensor) -> torch.Tensor:
        n_x = self.dataset.length_x
        n_y = self.dataset.length_y
        return n_x * n_y * statistic / (n_x + n_y)

    def compute_statistic(self) -> torch.Tensor:
        """computing MMD value for whole dataset.
        Note: if N(data) > batch_size, statistics is avg(mmd) over batch_size"""
        statistics = self.__compute_mmd()
        normalized_statistic = self.normalize_statistic(statistics)
        return normalized_statistic
'''