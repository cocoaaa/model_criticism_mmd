import typing
import torch
from typing import Union

import numpy as np

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel, KernelMatrixObject
from model_criticism_mmd.models.static import DEFAULT_DEVICE

FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')


class BasicRBFKernelFunction(BaseKernel):
    """A Kernel class with Euclidean-distance
    """
    def __init__(self,
                 device_obj: torch.device = DEFAULT_DEVICE,
                 log_sigma: Union[float, torch.Tensor] = -1.0,
                 opt_sigma: bool = False,
                 possible_shapes: typing.Tuple[int, ...] = (2,)):
        """
        Args:
            device_obj: torch.device object.
            log_sigma: sigma parameter of RBF kernel. This value should be log(sigma).
            Default is -1.0, which means median heuristic is used.
            opt_sigma: if True, then sigma parameter will be optimized during training. False, not.
        """
        super().__init__(device_obj=device_obj, possible_shapes=possible_shapes)
        self.device_obj = device_obj
        self.opt_sigma = opt_sigma
        if isinstance(log_sigma, torch.Tensor):
            logger.info(f'Given log_sigma is used. Check out if the configuration correspond to your intention. '
                        f'{log_sigma}')
            self.log_sigma = log_sigma
        elif isinstance(log_sigma, float):
            self.log_sigma = torch.tensor([log_sigma], requires_grad=opt_sigma, device=device_obj)
        else:
            raise TypeError('log_sigma should be tensor or float.')
        # for common use with base class
        self.lengthscale = self.log_sigma

    @classmethod
    def init_with_median(cls,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         scales: torch.Tensor,
                         opt_sigma: bool = False,
                         device_obj: torch.device = DEFAULT_DEVICE) -> "BasicRBFKernelFunction":
        """initialize RBFKernel with median heuristic.
        About the median heuristic, see https://arxiv.org/pdf/1707.07269.pdf

        Returns:
            RBFKernelFunction
        """
        # initialization of initial-sigma value
        rep_x = torch.mul(x, scales)
        rep_y = torch.mul(y, scales)
        __init_log_sigma = cls.get_median(x=rep_x, y=rep_y, is_log=True)
        return cls(
            device_obj=device_obj,
            log_sigma=__init_log_sigma,
            opt_sigma=opt_sigma)

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              log_sigma: torch.Tensor = None,
                              **kwargs) -> KernelMatrixObject:
        if log_sigma is None:
            log_sigma = self.log_sigma
        # end if
        sigma = torch.exp(log_sigma)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))

        # torch.t() is transpose function. torch.dot() is only for vectors. For 2nd tensors, "mm".
        xx = torch.mm(x, torch.t(x))
        xy = torch.mm(x, torch.t(y))
        yy = torch.mm(y, torch.t(y))

        x_sqnorms = torch.diagonal(xx, offset=0)
        y_sqnorms = torch.diagonal(yy, offset=0)

        k_xy = torch.exp(-1 * gamma * (-2 * xy + x_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :]))
        k_xx = torch.exp(-1 * gamma * (-2 * xx + x_sqnorms[:, np.newaxis] + x_sqnorms[np.newaxis, :]))
        k_yy = torch.exp(-1 * gamma * (-2 * yy + y_sqnorms[:, np.newaxis] + y_sqnorms[np.newaxis, :]))

        return KernelMatrixObject(k_xx=k_xx, k_yy=k_yy, k_xy=k_xy)

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, FloatOrTensor]:
        if is_grad_param_only and self.opt_sigma:
            __ = {'log_sigma': self.log_sigma}
        elif is_grad_param_only and self.opt_sigma is False:
            __ = {}
        else:
            __ = {'log_sigma': self.log_sigma}
        # end if
        return __

    def set_lengthscale(self, x: torch.Tensor, y: torch.Tensor, is_log: bool = True) -> None:
        log_median = self.get_median(x, y, is_log=is_log)
        self.lengthscale = log_median
        if self.opt_sigma is True:
            self.log_sigma = torch.tensor(log_median, device=self.device_obj, requires_grad=True)
        else:
            self.log_sigma = torch.tensor(log_median, device=self.device_obj)
        # end if
