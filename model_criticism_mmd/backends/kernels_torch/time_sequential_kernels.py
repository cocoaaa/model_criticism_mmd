import logging
import typing
import torch
from typing import Union

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import KernelMatrixObject
from model_criticism_mmd.backends.kernels_torch.rbf_kernel import BasicRBFKernelFunction
from model_criticism_mmd.supports.metrics.soft_dtw import SoftDTW

FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')



def func_compute_kernel_matrix(x: torch.Tensor,
                               y: torch.Tensor,
                               device_obj: torch.device,
                               gamma: float,
                               normalize: bool = False,
                               is_zero_minus_values: bool = True
                               ) -> torch.Tensor:
    soft_dtw_generator = SoftDTW(use_cuda=True if device_obj.type == 'cuda' else False,
                                 gamma=gamma, normalize=normalize)
    mat = torch.zeros((x.shape[0], y.shape[0]))
    for i_row in range(x.shape[0]):
        for i_col in range(i_row, y.shape[0]):
            x_sample = x[i_row]
            y_sample = y[i_col]
            x_tensor = torch.unsqueeze(x_sample, 0)
            y_tensor = torch.unsqueeze(y_sample, 0)
            __ = soft_dtw_generator.forward(x_tensor, y_tensor, is_return_matrix=False)
            mat[i_row, i_col] = __[0]
            # end if
        # end for
    # end for
    d_matrix = mat + mat.T - torch.diag(mat.diagonal())
    d_matrix = d_matrix.clamp(0, None)
    return d_matrix


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
                 log_sigma: Union[float, torch.Tensor] = 100,
                 normalize: bool = False,
                 device_obj: torch.device = device_default,
                 opt_sigma: bool = False,
                 possible_shapes=(3,)):
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
        self.log_sigma = torch.tensor([log_sigma]) if isinstance(log_sigma, float) else log_sigma
        self.opt_sigma = opt_sigma
        super().__init__(device_obj=device_obj, possible_shapes=possible_shapes, log_sigma=log_sigma)

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        if 'log_sigma' not in kwargs:
            log_sigma = self.log_sigma
        else:
            log_sigma = kwargs['log_sigma']
        # end if
        d_xx = func_compute_kernel_matrix(x, x, device_obj=self.device_obj, gamma=self.gamma,
                                          normalize=self.normalize)
        d_yy = func_compute_kernel_matrix(y, y, device_obj=self.device_obj, gamma=self.gamma,
                                          normalize=self.normalize)
        d_xy = func_compute_kernel_matrix(x, y, device_obj=self.device_obj, gamma=self.gamma,
                                          normalize=self.normalize)
        gamma = torch.div(1, (2 * torch.pow(log_sigma, 2)))
        k_xx = torch.exp(-1 * gamma * torch.pow(d_xx, 2))
        k_yy = torch.exp(-1 * gamma * torch.pow(d_yy, 2))
        k_xy = torch.exp(-1 * gamma * torch.pow(d_xy, 2))

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
