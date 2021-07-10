import numpy
import numpy as np
import torch
import typing
from torch.utils.data import Dataset
from torch import Tensor
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.models import TrainingLog, TrainedMmdParameters, TrainerBase, MmdValues, TypeInputData, \
    TwoSampleDataSet
from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd.exceptions import NanException
import gc

device_default = torch.device('cpu')


class MMD(object):
    def __init__(self,
                 kernel_function_obj: kernels_torch.BaseKernel,
                 scales: typing.Optional[torch.Tensor] = None,
                 device_obj: torch.device = device_default,
                 biased: bool = True):
        """
        Args:
            kernel_function_obj: Kernel function by your preference.
            scales: A vector to scales x and y. A scaling operation is elementwise-product.
            If None, no operations of elementwise-product.
            device_obj: device object of torch.
            biased: If True, then MMD estimator is biased estimator. Else unbiased estimator.
        """
        self.kernel_function_obj = kernel_function_obj
        self.device_obj = device_obj
        self.min_var_est = torch.tensor([1e-8], dtype=torch.float64, device=device_obj)
        self.scales = scales
        self.biased = biased

    def _mmd2_and_variance(self,
                           k_xx: torch.Tensor,
                           k_xy: torch.Tensor,
                           k_yy: torch.Tensor, unit_diagonal=False):
        m = k_xx.shape[0]  # Assumes X, Y are same shape

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if unit_diagonal:
            diag_x = diag_y = 1
            sum_diag_x = sum_diag_y = m
            sum_diag2_x = sum_diag2_y = m
        else:
            diag_x = torch.diagonal(k_xx)
            diag_y = torch.diagonal(k_yy)

            sum_diag_x = diag_x.sum()
            sum_diag_y = diag_y.sum()

            sum_diag2_x = diag_x.dot(diag_x)
            sum_diag2_y = diag_y.dot(diag_y)
        # end if
        # Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        kt_xx_sums = torch.sum(k_xx, dim=1) - diag_x
        # Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        kt_yy_sums = torch.sum(k_yy, dim=1) - diag_y
        # K_XY_sums_0 = K_XY.sum(axis=0)
        k_xy_sums_0 = torch.sum(k_xy, dim=0)
        # K_XY_sums_1 = K_XY.sum(axis=1)
        k_xy_sums_1 = torch.sum(k_xy, dim=1)

        kt_xx_sum = kt_xx_sums.sum()
        kt_yy_sum = kt_yy_sums.sum()
        k_xy_sum = k_xy_sums_0.sum()

        kt_xx_2_sum = (k_xx ** 2).sum() - sum_diag2_x
        kt_yy_2_sum = (k_yy ** 2).sum() - sum_diag2_y
        k_xy_2_sum = (k_xy ** 2).sum()

        if self.biased:
            mmd2 = ((kt_xx_sum + sum_diag_x) / (m * m) + (kt_yy_sum + sum_diag_y) / (m * m)
                    - 2 * k_xy_sum / (m * m))
        else:
            mmd2 = (kt_xx_sum / (m * (m-1)) + kt_yy_sum / (m * (m-1)) - 2 * k_xy_sum / (m * m))
        # end if

        var_est = (
              2 / (m**2 * (m-1)**2) * (
                  2 * kt_xx_sums.dot(kt_xx_sums) - kt_xx_2_sum
                  + 2 * kt_yy_sums.dot(kt_yy_sums) - kt_yy_2_sum)
              - (4*m-6) / (m**3 * (m-1)**3) * (kt_xx_sum**2 + kt_yy_sum**2)
              + 4*(m-2) / (m**3 * (m-1)**2) * (
                  k_xy_sums_1.dot(k_xy_sums_1) + k_xy_sums_0.dot(k_xy_sums_0))
              - 4 * (m-3) / (m**3 * (m-1)**2) * k_xy_2_sum
              - (8*m - 12) / (m**5 * (m-1)) * k_xy_sum**2
              + 8 / (m**3 * (m-1)) * (
                  1/m * (kt_xx_sum + kt_yy_sum) * k_xy_sum
                  - kt_xx_sums.dot(k_xy_sums_1)
                  - kt_yy_sums.dot(k_xy_sums_0))
        )

        return mmd2, var_est

    def _mmd2_and_ratio(self,
                        k_xx: torch.Tensor,
                        k_xy: torch.Tensor,
                        k_yy: torch.Tensor,
                        unit_diagonal: bool = False,
                        min_var_est: typing.Optional[torch.Tensor] = None
                        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if min_var_est is None:
            __min_var_est = self.min_var_est
        else:
            __min_var_est = min_var_est
        # end if
        mmd2, var_est = self._mmd2_and_variance(k_xx, k_xy, k_yy, unit_diagonal=unit_diagonal)
        ratio = torch.div(mmd2, torch.sqrt(torch.max(var_est, __min_var_est)))
        return mmd2, ratio

    def process_mmd2_and_ratio(self,
                               x: torch.Tensor,
                               y: torch.Tensor,
                               **kwargs
                               ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Computes MMD value.

        Returns:
            tuple: (mmd_value, ratio)
        """
        kernel_matrix_obj = self.kernel_function_obj.compute_kernel_matrix(x=x, y=y, **kwargs)
        return self._mmd2_and_ratio(kernel_matrix_obj.k_xx, kernel_matrix_obj.k_xy, kernel_matrix_obj.k_yy,
                                    unit_diagonal=True)

    @staticmethod
    def operation_scale_product(scales: torch.Tensor,
                                input_p: torch.Tensor,
                                input_q: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Element-wise product. scales(1d tensor) * input_{p, q}(2d tensor)
        Note: if input_{p, q} have different dimensions, we cut scales into the same size as input dimensions.

        Args:
            scales: 1d tensor
            input_p: 2d tensor
            input_q: 2d tensor

        Returns: (2d tensor, 2d tensor)
        """
        if input_p.shape[-1] != input_q.shape[-1]:
            scales_p = scales[0:input_p.shape[-1]]
            scales_q = scales[0:input_q.shape[-1]]
            rep_p = torch.mul(scales_p, input_p)
            rep_q = torch.mul(scales_q, input_q)
        else:
            rep_p = torch.mul(scales, input_p)
            rep_q = torch.mul(scales, input_q)

        return rep_p, rep_q

    def mmd_distance(self,
                     x: TypeInputData,
                     y: TypeInputData,
                     is_detach: bool = False,
                     **kwargs) -> MmdValues:
        """Computes MMD value.

        Returns:
            MmdValues: named tuple object.
        """
        self.kernel_function_obj.check_data_shape(x)
        self.kernel_function_obj.check_data_shape(y)

        if self.scales is not None:
            __x = torch.tensor(x) if isinstance(x, numpy.ndarray) else x
            __y = torch.tensor(y) if isinstance(x, numpy.ndarray) else y
            rep_x, rep_y = self.operation_scale_product(self.scales, __x, __y)
        else:
            __x = torch.tensor(x) if isinstance(x, numpy.ndarray) else x
            __y = torch.tensor(y) if isinstance(x, numpy.ndarray) else y
            rep_x, rep_y = __x, __y
        # end if
        mmd2, ratio = self.process_mmd2_and_ratio(rep_x, rep_y, **kwargs)
        if is_detach:
            return MmdValues(mmd2.cpu().detach(), ratio.cpu().detach())
        else:
            return MmdValues(mmd2, ratio)


class ModelTrainerTorchBackend(TrainerBase):
    """A class to optimize MMD."""
    def __init__(self,
                 mmd_estimator: MMD,
                 device_obj: torch.device = device_default):
        self.mmd_estimator = mmd_estimator
        self.device_obj = device_obj
        self.obj_value_min_threshold = torch.tensor([1e-6], device=self.device_obj)
        self.default_reg = torch.tensor([0], device=self.device_obj)

    @classmethod
    def model_from_trained(cls,
                           parameter_obj: TrainedMmdParameters,
                           device_obj: torch.device = device_default) -> "ModelTrainerTorchBackend":
        """returns ModelTrainerTorchBackend instance from the trained-parameters."""
        scales = torch.tensor(parameter_obj.scales, device=device_obj)
        model_obj = cls(mmd_estimator=MMD(kernel_function_obj=parameter_obj.kernel_function_obj,
                                          scales=scales,
                                          device_obj=device_obj), device_obj=device_obj)
        return model_obj

    def init_scales(self,
                    size_dimension: int,
                    init_scale: torch.Tensor) -> torch.Tensor:
        """A scale vector which scales the input matrix X.
        must be the same size as the input data."""

        if init_scale is None:
            scales: torch.Tensor = torch.rand(size=(size_dimension,), requires_grad=True, device=self.device_obj)
        else:
            logger.info('Set the initial scales value')
            scales = torch.tensor(init_scale.clone().detach().cpu(), requires_grad=True, device=self.device_obj)
        # end if
        return scales

    def run_validation(self,
                       dataset_validation: TwoSampleDataSet,
                       reg: torch.Tensor,
                       batchsize: int,
                       num_workers: int,
                       is_shuffle: bool = False,
                       is_validation_all: bool = False,
                       is_first_iter: bool = False,
                       is_gc: bool = False
                       ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A procedure for validations
        """
        if (dataset_validation.length_x < batchsize) and ((dataset_validation.length_y < batchsize)):
            logger.debug(f'in validation step, is_validation_all is True. '
                         f'N(x)={dataset_validation.length_x},N(y)={dataset_validation.length_y} < batchsize')
            is_validation_all = True
        # end if

        if is_validation_all:
            x_val, y_val = dataset_validation.get_all_item()
            val_mmd2_pq, val_stat, val_obj = self.forward(x_val, y_val, reg=reg, is_validation=True)
            if is_first_iter:
                logger.info(
                    f'Validation at 0. MMD^2 = {val_mmd2_pq.detach().cpu().numpy()}, '
                    f'ratio = {val_stat.detach().cpu().numpy()} '
                    f'obj = {val_obj.detach().cpu().numpy()}')
            return val_mmd2_pq, val_obj, val_stat
        else:
            total_mmd2_val = 0
            total_obj_val = 0
            total_stat_val = 0
            n_batches = 0
            if self.device_obj == torch.device('cpu'):
                data_loader = torch.utils.data.DataLoader(dataset_validation,
                                                          batch_size=batchsize, shuffle=is_shuffle,
                                                          num_workers=num_workers)
            else:
                data_loader = torch.utils.data.DataLoader(dataset_validation,
                                                          batch_size=batchsize, shuffle=is_shuffle)
            # end if
            for xbatch, ybatch in data_loader:
                mmd2_pq, stat, obj = self.forward(xbatch, ybatch, reg=reg)
                # end if
                assert np.isfinite(mmd2_pq.detach().cpu().numpy())
                assert np.isfinite(obj.detach().cpu().numpy())
                total_mmd2_val += mmd2_pq
                total_obj_val += obj
                total_stat_val += stat
                n_batches += 1
                if is_gc:
                    del mmd2_pq, obj, stat
                    del xbatch, ybatch
                    gc.collect()
                # end if
            # end for
            avg_mmd2 = torch.div(total_mmd2_val, n_batches)
            avg_obj = torch.div(total_obj_val, n_batches)
            avg_stat = torch.div(total_stat_val, n_batches)

            if is_first_iter:
                logger.info(
                    f'Validation(mean over batch) at 0. '
                    f'MMD^2 = {avg_mmd2.detach().cpu().numpy()}, '
                    f'ratio = {avg_stat.detach().cpu().numpy()} '
                    f'obj = {avg_obj.detach().cpu().numpy()}')
            return avg_mmd2, avg_obj, avg_stat

    def run_train_epoch(self,
                        optimizer: torch.optim.SGD,
                        dataset: TwoSampleDataSet,
                        batchsize: int,
                        reg: torch.Tensor,
                        num_workers: int = 1,
                        is_scales_non_negative: bool = False,
                        is_shuffle: bool = False,
                        is_gc: bool = False
                        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        total_mmd2 = 0
        total_obj = 0
        n_batches = 0

        if self.device_obj == torch.device('cpu'):
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=is_shuffle,
                                                      num_workers=num_workers)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=is_shuffle)
        # end if
        for xbatch, ybatch in data_loader:
            optimizer.zero_grad()
            mmd2_pq, stat, obj = self.forward(xbatch, ybatch, reg=reg)
            # end if
            assert np.isfinite(mmd2_pq.detach().cpu().numpy())
            assert np.isfinite(obj.detach().cpu().numpy())
            total_mmd2 += mmd2_pq
            total_obj += obj
            n_batches += 1
            # do differentiation now.
            obj.backward()
            #
            optimizer.step()
            if is_gc:
                del mmd2_pq, obj, stat
                del xbatch, ybatch
                gc.collect()
            # end if
            if len(self.scales[torch.isnan(self.scales)]):
                raise NanException('scales vector goes into Nan. Stop training.')
            # end if
            if is_scales_non_negative:
                with torch.no_grad():
                    self.scales[:] = self.scales.clamp(0, None)
                # end with
            # end if
        # end for
        avg_mmd2 = torch.div(total_mmd2, n_batches)
        avg_obj = torch.div(total_obj, n_batches)

        return avg_mmd2, avg_obj

    # ----------------------------------------------------------------------

    def forward(self,
                input_p: torch.Tensor,
                input_q: torch.Tensor,
                reg: typing.Optional[Tensor] = None,
                opt_log: bool = True,
                is_validation: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A procedure to compute mmd-value with the given parameters.
        The procedure correspond to forward() operation.

        Args:
            input_p: data
            input_q: data
            reg: a term for regularization
            opt_log: a flag to control heuristic lower bound for the objective value.
            is_validation:
        Returns:
            tuple: mmd2_pq, stat, obj
        """
        if reg is None:
            reg__ = self.default_reg
        else:
            reg__ = reg
        # end if
        # 1. elementwise-product(data, scale-vector)
        mmd2_pq, stat = self.mmd_estimator.mmd_distance(input_p, input_q, is_detach=False, is_validation=is_validation)
        # 2. define the objective-value
        obj = -(torch.log(torch.max(stat, self.obj_value_min_threshold)) if opt_log else stat) + reg__
        return mmd2_pq, stat, obj

    @staticmethod
    def log_message(epoch: int, avg_mmd2: Tensor, avg_obj: Tensor, val_mmd2_pq: Tensor,
                    val_stat: Tensor, val_obj: Tensor):
        fmt = ("{: >6,}: [avg train] MMD^2 {} obj {} "
               "val-MMD^2 {} val-ratio {} val-obj {}  elapsed: {:,}")
        if epoch in {0, 5, 25, 50} or epoch % 100 == 0:
            logger.info(
                fmt.format(epoch, avg_mmd2.detach().cpu().numpy(),
                           avg_obj.detach().cpu().numpy(),
                           val_mmd2_pq.detach().cpu().numpy(),
                           val_stat.detach().cpu().numpy(),
                           val_obj.detach().cpu().numpy(),
                           0.0))
        # end if

    def train(self,
              dataset_training: TwoSampleDataSet,
              dataset_validation: TwoSampleDataSet,
              num_epochs: int = 1000,
              batchsize: int = 200,
              ratio_train: float = 0.8,
              reg: typing.Optional[torch.Tensor] = None,
              initial_scale: torch.Tensor = None,
              lr: float = 0.01,
              opt_log: bool = True,
              num_workers: int = 1,
              is_scales_non_negative: bool = False,
              is_training_auto_stop: bool = False,
              auto_stop_epochs: int = 10,
              auto_stop_threshold: float = 0.00001,
              is_shuffle: bool = False,
              is_validation_all: bool = False,
              is_gc: bool = False) -> TrainedMmdParameters:
        """Training (Optimization) of MMD parameters.

        Args:
            dataset_training:
            num_epochs: #epochs.
            batchsize: batch size
            ratio_train: a ratio of division for training
            reg:
            initial_scale: initial value of scales vector. If None, the vector is initialized randomly.
            lr: learning rate
            opt_log: flag to control training procedure. If True, then objective-value has lower-bound. else Not.
            dataset_validation:
            num_workers: #worker for training.
            is_scales_non_negative: if True then scales set non-negative. if False, no control.
            is_training_auto_stop: if True, then training is auto-stopped. if False, no auto-stop.
            auto_stop_epochs: If epoch=1 (auto-stop), the epoch size that training is auto stopped.
            When objective values are constant in auto_stop_epochs, the training-procedure is auto-stopped.
            auto_stop_threshold: The threshold to stop trainings automatically.
            is_shuffle: Dataset will be selected randomly or NOT. if True, then sample is selected randomly.
            if False, selected sequentially.
            is_validation_all: True, if you'd like to run validations with batch=1.
            False, then val. values will be averaged with the same batch-size of a training.
            is_gc: True if you release memory after each batch, False no.
            Normally, the speed will be slower if is_gc=True. You use the option when memory leaks during trainings.
        Returns:
            TrainedMmdParameters
        """
        assert num_epochs > 0
        assert dataset_training.get_dimension() == dataset_validation.get_dimension()

        dimension_longer: int = dataset_training.get_dimension()[0]
        self.scales = self.init_scales(size_dimension=dimension_longer, init_scale=initial_scale)
        self.mmd_estimator.scales = self.scales

        # collects parameters to be optimized / set an optimizer
        kernel_params_target = self.mmd_estimator.kernel_function_obj.get_params(is_grad_param_only=True)
        params_target = [self.scales] + list(kernel_params_target.values())
        optimizer = torch.optim.SGD(params_target, lr=lr, momentum=0.9, nesterov=True)

        self.run_validation(dataset_validation, reg, batchsize, num_workers, is_shuffle, is_validation_all,
                            is_first_iter=True, is_gc=is_gc)
        # procedure of trainings
        training_log = []
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            avg_mmd2, avg_obj = self.run_train_epoch(optimizer,
                                                     dataset_training,
                                                     batchsize=batchsize,
                                                     reg=reg,
                                                     num_workers=num_workers,
                                                     is_scales_non_negative=is_scales_non_negative,
                                                     is_shuffle=is_shuffle,
                                                     is_gc=is_gc)
            val_mmd2_pq, val_obj, val_stat = self.run_validation(dataset_validation=dataset_validation,
                                                                 reg=reg,
                                                                 batchsize=batchsize,
                                                                 num_workers=num_workers,
                                                                 is_shuffle=is_shuffle,
                                                                 is_validation_all=is_validation_all,
                                                                 is_gc=is_gc)
            training_log.append(TrainingLog(epoch=epoch,
                                            avg_mmd_training=avg_mmd2.detach().cpu().numpy(),
                                            avg_obj_train=avg_obj.detach().cpu().numpy(),
                                            mmd_validation=val_mmd2_pq.detach().cpu().numpy(),
                                            obj_validation=val_obj.detach().cpu().numpy(),
                                            sigma=None,
                                            scales=self.scales.detach().cpu().numpy()))
            self.log_message(epoch, avg_mmd2, avg_obj, val_mmd2_pq, val_stat, val_obj)
            if is_training_auto_stop and len(training_log) > auto_stop_epochs:
                __val_validations = [t_obj.obj_validation for t_obj in training_log[epoch - auto_stop_epochs:epoch]]
                __variance_validations = max(__val_validations) - min(__val_validations)
                if __variance_validations < auto_stop_threshold:
                    logger.info(f'Training stops at {epoch} automatically because epoch is set '
                                f'and variance in {auto_stop_epochs} '
                                f'epochs are within {__variance_validations} < {auto_stop_threshold}')
                    break
                # end if
            # end if
        # end for
        return TrainedMmdParameters(
            scales=self.scales.detach().cpu().numpy(),
            training_log=training_log,
            kernel_function_obj=self.mmd_estimator.kernel_function_obj)

    def mmd_distance(self,
                     x: TypeInputData,
                     y: TypeInputData,
                     is_detach: bool = False) -> MmdValues:
        return self.mmd_estimator.mmd_distance(x, y, is_detach)
