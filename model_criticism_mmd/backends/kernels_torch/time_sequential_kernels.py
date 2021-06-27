import logging
import typing
import torch
from torch.nn.functional import normalize
from typing import Union

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel, KernelMatrixObject
from model_criticism_mmd.backends.kernels_torch.rbf_kernel import BasicRBFKernelFunction
from model_criticism_mmd.supports.metrics.soft_dtw import SoftDTW

FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')


class SoftDtwKernelFunctionTimeSample(BasicRBFKernelFunction):
    """A Kernel class when your data is temporal data.
    Your data is matrix form, of which a sample is data at t=i.

    In a column-direction, values show data "feature" of t=i.
    In a row-direction, values show data "sample" of t=i.

    In this class, distance-metric is an alignment cost of SoftDTW algorithm.
    The alignment-cost is represented with matrix R which correspond to intermediary alignment costs.

    The kernel is RBFKernel, thus the Eq of the class is
    RbfKernel(x, y) = exp(-1 * (alignment-cost) / sigma)
    """
    def __init__(self,
                 gamma: float = 1.0,
                 log_sigma: Union[float, torch.Tensor] = 0.0,
                 normalize: bool = False,
                 device_obj: torch.device = device_default,
                 opt_sigma: bool = False):
        """

        Args:
            gamma: a parameter of SoftDTW
            log_sigma:
            normalize:
            device_obj:
            opt_sigma:
        """
        self.gamma = gamma
        self.normalize = normalize
        self.log_sigma = torch.tensor(log_sigma) if isinstance(log_sigma, float) else log_sigma
        self.soft_dtw = SoftDTW(use_cuda=True if device_obj.type == 'cuda' else False, gamma=gamma, normalize=False)
        self.opt_sigma = opt_sigma
        super().__init__(device_obj=device_obj)

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        if 'is_validation' in kwargs and kwargs['is_validation'] is True:
            # if phrase is in validation, sample-size might be different. Set validation sample with the small-size.
            validation_smaller = min(x.shape[0], y.shape[0])
            x = x[:validation_smaller, :]
            y = y[:validation_smaller, :]
        # end if

        if len(x[torch.isnan(x)]) > 0:
            x_input = x[~torch.any(torch.isnan(x), dim=1)]
            logger.debug('Deleted padding samples')
            assert len(x_input) > 0, f'{x_input.shape}'
        else:
            x_input = x
        if len(y[torch.isnan(y)]) > 0:
            y_input = y[~torch.any(torch.isnan(y), dim=1)]
            logger.debug('Deleted padding samples')
            assert len(y_input) > 0, f'{y_input.shape}'
        else:
            y_input = y
        # end if

        if 'log_sigma' not in kwargs:
            log_sigma = self.log_sigma
        else:
            log_sigma = kwargs['log_sigma']
        # end if
        sigma = torch.exp(log_sigma)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        __x = x_input.unsqueeze(0) if len(x_input.size()) == 2 else x
        __y = y_input.unsqueeze(0) if len(y_input.size()) == 2 else y

        soft_dt_xx = torch.pow(self.soft_dtw.forward(__x, __x, is_return_matrix=True), 2)
        soft_dt_xy = torch.pow(self.soft_dtw.forward(__x, __y, is_return_matrix=True), 2)
        soft_dt_yy = torch.pow(self.soft_dtw.forward(__y, __y, is_return_matrix=True), 2)

        if self.normalize:
            soft_dt_xx = normalize(soft_dt_xx, dim=0)
            soft_dt_xy = normalize(soft_dt_xy, dim=0)
            soft_dt_yy = normalize(soft_dt_yy, dim=0)
        # end if

        k_xx = torch.exp(-1 * gamma * soft_dt_xx).squeeze(0)
        k_yy = torch.exp(-1 * gamma * soft_dt_yy).squeeze(0)
        k_xy = torch.exp(-1 * gamma * soft_dt_xy).squeeze(0)

        return KernelMatrixObject(k_xx, k_yy, k_xy)

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, torch.Tensor]:
        if is_grad_param_only:
            return {'log_sigma': self.log_sigma}
        else:
            return {'gamma': self.gamma, 'log_sigma': self.log_sigma, 'normalize': self.normalize}


# class SoftDtwKernelFunctionTimeFeature(BasicRBFKernelFunction):
#     """A Kernel class when your data is temporal data.
#     Your data is matrix form, of which  each feature represents t=i.
#
#     In a column-direction, values show data "sample"..
#     In a row-direction, values show data "feature" of t=i.
#
#     In this class, distance-metric is score by SoftDTW algorithm.
#
#     The kernel is RBFKernel, thus the Eq of the class is
#     RbfKernel(x, y) = exp(-1 * (softDTW(x, y)) / sigma)
#     """
#     pass
