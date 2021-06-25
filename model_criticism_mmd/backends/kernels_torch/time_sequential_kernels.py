import typing
import torch
from typing import Union

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel, KernelMatrixObject
from sdtw.distance import SquaredEuclidean
from sdtw.soft_dtw import SoftDTW


FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')


class SoftDtwKernelFunctionTimeSample(BaseKernel):
    """A Kernel class when your data is temporal data.
    Your data is matrix form, of which a sample is data at t=i.

    In a column-direction, values show data "feature" of t=i.
    In a row-direction, values show data "sample" of t=i."""
    def __init__(self,
                 gamma: float = 1.0,
                 log_sigma: Union[float, torch.Tensor] = 0.0,
                 device_obj: torch.device = device_default):
        self.gamma = gamma
        self.log_sigma = torch.tensor(log_sigma) if isinstance(log_sigma, float) else log_sigma
        super().__init__(device_obj=device_obj)

    def compute_soft_dtw(self, x: torch.Tensor, y: torch.Tensor):
        # D can also be an arbitrary distance matrix: numpy array, shape [m, n]
        matrix_d = SquaredEuclidean(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        sdtw = SoftDTW(matrix_d, gamma=self.gamma)
        # soft-DTW discrepancy, approaches DTW as gamma -> 0
        value = sdtw.compute()
        # gradient w.r.t. D, shape = [m, n], which is also the expected alignment matrix
        matrix_e = sdtw.grad()
        # gradient w.r.t. X, shape = [m, d]
        # matrix_g = matrix_d.jacobian_product(matrix_e)

        return torch.tensor(matrix_e, device=self.device_obj)

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        if 'log_sigma' not in kwargs:
            log_sigma = self.log_sigma
        else:
            log_sigma = kwargs['log_sigma']
        # end if
        sigma = torch.exp(log_sigma)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        soft_dt_xx = self.compute_soft_dtw(x, x)
        soft_dt_xy = self.compute_soft_dtw(x, y)
        soft_dt_yy = self.compute_soft_dtw(y, y)

        k_xx = torch.exp(-1 * gamma * soft_dt_xx)
        k_yy = torch.exp(-1 * gamma * soft_dt_yy)
        k_xy = torch.exp(-1 * gamma * soft_dt_xy)

        return KernelMatrixObject(k_xx, k_yy, k_xy)

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, torch.Tensor]:
        if is_grad_param_only:
            return {}
        else:
            return {'gamma': self.gamma}
