import typing
import torch
from typing import Union

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel, KernelMatrixObject
from model_criticism_mmd.supports.metrics.soft_dtw import SoftDTW

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
        self.soft_dtw = SoftDTW(use_cuda=True if device_obj.type == 'cuda' else False, gamma=gamma, normalize=False)
        super().__init__(device_obj=device_obj)

    def compute_soft_dtw(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        matrix_alignment = self.soft_dtw.forward(x, y, is_return_matrix=True)
        return matrix_alignment

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        if 'log_sigma' not in kwargs:
            log_sigma = self.log_sigma
        else:
            log_sigma = kwargs['log_sigma']
        # end if
        sigma = torch.exp(log_sigma)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        __x = x.unsqueeze(0) if len(x.size()) == 2 else x
        __y = y.unsqueeze(0) if len(y.size()) == 2 else y

        soft_dt_xx = self.compute_soft_dtw(__x, __x)
        soft_dt_xy = self.compute_soft_dtw(__x, __y)
        soft_dt_yy = self.compute_soft_dtw(__y, __y)

        k_xx = torch.exp(-1 * gamma * soft_dt_xx).squeeze(0)
        k_yy = torch.exp(-1 * gamma * soft_dt_yy).squeeze(0)
        k_xy = torch.exp(-1 * gamma * soft_dt_xy).squeeze(0)

        return KernelMatrixObject(k_xx, k_yy, k_xy)

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, torch.Tensor]:
        if is_grad_param_only:
            return {}
        else:
            return {'gamma': self.gamma}
