import numpy
import numpy as np
import torch
import typing
from torch.utils.data import Dataset
from torch import Tensor
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.models import TrainingLog, TrainedMmdParameters, TrainerBase, MmdValues, TypeInputData
from model_criticism_mmd.backends import kernels_torch

device_default = torch.device('cpu')


class TwoSampleDataSet(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.length = len(x)
        assert len(x) == len(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


# ----------------------------------------------------------------------
# MMD equation

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
        kernel_matrix_obj = self.kernel_function_obj.compute_kernel_matrix(x=x, y=y)
        return self._mmd2_and_ratio(kernel_matrix_obj.k_xx, kernel_matrix_obj.k_xy, kernel_matrix_obj.k_yy,
                                    unit_diagonal=True)

    @staticmethod
    def operation_scale_product(scales: torch.Tensor,
                                input_p: torch.Tensor,
                                input_q: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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

        if self.scales is not None:
            assert len(self.scales) == x.shape[1] == y.shape[1],\
            f'Error at scales vector. Dimension size does not match. ' \
            f'The given scales {len(self.scales)}dims. x {x.shape[1]}dims. y {y.shape[1]}dims.'
            __x = torch.tensor(x) if isinstance(x, numpy.ndarray) else x
            __y = torch.tensor(y) if isinstance(x, numpy.ndarray) else y
            rep_x, rep_y = self.operation_scale_product(self.scales, __x, __y)
        else:
            __x = torch.tensor(x) if isinstance(x, numpy.ndarray) else x
            __y = torch.tensor(y) if isinstance(x, numpy.ndarray) else y
            rep_x, rep_y = __x, __y
        # end if
        mmd2, ratio = self.process_mmd2_and_ratio(rep_x, rep_y)
        if is_detach:
            return MmdValues(mmd2.cpu().detach(), ratio.cpu().detach())
        else:
            return MmdValues(mmd2, ratio)


# ----------------------------------------------------------------------


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

    def init_scales(self, data: torch.Tensor, init_scale: torch.Tensor) -> torch.Tensor:
        """A scale vector which scales the input matrix X.
        must be the same size as the input data."""

        if init_scale is None:
            scales: torch.Tensor = torch.rand(size=(data.shape[1],), requires_grad=True, device=self.device_obj)
        else:
            logger.info('Set the initial scales value')
            assert data.shape[1] == init_scale.shape[0]
            scales = torch.tensor(init_scale.clone().detach().cpu(), requires_grad=True, device=self.device_obj)
        # end if
        return scales

    def run_train_epoch(self,
                        optimizer: torch.optim.SGD,
                        dataset: TwoSampleDataSet,
                        batchsize: int,
                        reg: torch.Tensor,
                        num_workers: int = 1,
                        is_scales_non_negative: bool = False
                        ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        total_mmd2 = 0
        total_obj = 0
        n_batches = 0

        if self.device_obj == torch.device('cpu'):
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True,
                                                      num_workers=num_workers)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
        # end if
        for xbatch, ybatch in data_loader:
            mmd2_pq, stat, obj = self.forward(xbatch, ybatch, reg=reg)
            assert np.isfinite(mmd2_pq.detach().cpu().numpy())
            assert np.isfinite(obj.detach().cpu().numpy())
            total_mmd2 += mmd2_pq
            total_obj += obj
            n_batches += 1
            # do differentiation now.
            obj.backward()
            #
            optimizer.step()
            optimizer.zero_grad()
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
                opt_log: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A procedure to compute mmd-value with the given parameters.
        The procedure correspond to forward() operation.

        Args:
            input_p: data
            input_q: data
            reg: a term for regularization
            opt_log: a flag to control heuristic lower bound for the objective value.
        Returns:
            tuple: mmd2_pq, stat, obj
        """
        if reg is None:
            reg__ = self.default_reg
        else:
            reg__ = reg
        # end if
        # 1. elementwise-product(data, scale-vector)
        mmd2_pq, stat = self.mmd_estimator.mmd_distance(input_p, input_q)
        # 2. define the objective-value
        obj = -(torch.log(torch.max(stat, self.obj_value_min_threshold)) if opt_log else stat) + reg__
        return mmd2_pq, stat, obj

    @staticmethod
    def to_tensor(data: np.ndarray) -> Tensor:
        if isinstance(data, np.ndarray):
            return torch.tensor(data)
        elif isinstance(data, Tensor):
            return data
        else:
            raise TypeError()

    def split_data(self,
                   x: TypeInputData,
                   y: TypeInputData,
                   x_val: typing.Optional[TypeInputData],
                   y_val: typing.Optional[TypeInputData],
                   ratio_train: float = 0.8
                   ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # data conversion
        x__ = self.to_tensor(x)
        y__ = self.to_tensor(y)

        if ratio_train < 1.0:
            __split_index = int(len(x) * ratio_train)
            x_train__ = x__[:__split_index]
            x_val__ = x__[__split_index:]
            y_train__ = y__[:__split_index]
            y_val__ = y__[__split_index:]
        else:
            x_train__ = x__
            y_train__ = y__
            x_val__ = self.to_tensor(x_val)
            y_val__ = self.to_tensor(y_val)
        # end if
        x_train__d = x_train__.to(self.device_obj)
        y_train__d = y_train__.to(self.device_obj)
        x_val__d = x_val__.to(self.device_obj)
        y_val__d = y_val__.to(self.device_obj)

        return x_train__d, y_train__d, x_val__d, y_val__d

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
              x_train: TypeInputData,
              y_train: TypeInputData,
              num_epochs: int = 1000,
              batchsize: int = 200,
              ratio_train: float = 0.8,
              reg: typing.Optional[torch.Tensor] = None,
              initial_scale: torch.Tensor = None,
              lr: float = 0.01,
              opt_log: bool = True,
              x_val: TypeInputData = None,
              y_val: TypeInputData = None,
              num_workers: int = 1,
              is_scales_non_negative: bool = False) -> TrainedMmdParameters:
        """Training (Optimization) of MMD parameters.

        Args:
            x_train: data
            y_train: data
            num_epochs: #epochs
            batchsize: batch size
            ratio_train: a ratio of division for training
            reg:
            initial_scale: initial value of scales vector. If None, the vector is initialized randomly.
            lr: learning rate
            opt_log: flag to control training procedure. If True, then objective-value has lower-bound. else Not.
            x_val: data for validation. If None, the data is picked from x_train.
            y_val: same as x_val.
            num_workers: #worker for training.
            is_scales_non_negative: if True then scales set non-negative. if False, no control.

        Returns:

        """
        # todo epoch auto-stop.

        assert len(x_train.shape) == len(y_train.shape) == 2
        logger.debug(f'input data N(sample-size)={x_train.shape[0]}, N(dimension)={x_train.shape[1]}')

        if x_val is None or y_val is None:
            x_train__, y_train__, x_val__, y_val__ = self.split_data(x_train, y_train, None, None, ratio_train)
        else:
            x_train__, y_train__, x_val__, y_val__ = self.split_data(x_train, y_train, x_val, y_val, 1.0)
        # end if
        self.scales = self.init_scales(data=x_train__, init_scale=initial_scale)
        self.mmd_estimator.scales = self.scales

        # collects parameters to be optimized / set an optimizer
        kernel_params_target = self.mmd_estimator.kernel_function_obj.get_params(is_grad_param_only=True)
        params_target = [self.scales] + list(kernel_params_target.values())
        optimizer = torch.optim.SGD(params_target, lr=lr, momentum=0.9, nesterov=True)
        # procedure of trainings
        dataset_train = TwoSampleDataSet(x_train__, y_train__)
        val_mmd2_pq, val_stat, val_obj = self.forward(x_val__, y_val__, reg=reg)
        logger.debug(
            f'Validation at 0. MMD^2 = {val_mmd2_pq.detach().cpu().numpy()}, '
            f'ratio = {val_stat.detach().cpu().numpy()} '
            f'obj = {val_obj.detach().cpu().numpy()}')

        training_log = []
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            avg_mmd2, avg_obj = self.run_train_epoch(optimizer,
                                                     dataset_train,
                                                     batchsize=batchsize,
                                                     reg=reg,
                                                     num_workers=num_workers,
                                                     is_scales_non_negative=is_scales_non_negative)
            val_mmd2_pq, val_stat, val_obj = self.forward(x_val__, y_val__, reg=reg)
            training_log.append(TrainingLog(epoch,
                                            avg_mmd2.detach().cpu().numpy(),
                                            avg_obj.detach().cpu().numpy(),
                                            val_mmd2_pq.detach().cpu().numpy(),
                                            val_obj.detach().cpu().numpy(),
                                            sigma=None,
                                            scales=self.scales.detach().cpu().numpy()))
            self.log_message(epoch, avg_mmd2, avg_obj, val_mmd2_pq, val_stat, val_obj)
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
