import torch
import numpy as np
import dataclasses
from typing import Union
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.kernels.matern_kernel import MaternKernel


@dataclasses.dataclass
class KernelMatrixObject(object):
    k_xx: Union[torch.Tensor, LazyEvaluatedKernelTensor]
    k_yy: Union[torch.Tensor, LazyEvaluatedKernelTensor]
    k_xy: torch.Tensor


class BaseKernel(object):
    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        raise NotImplementedError()


class RBFKernelFunction(BaseKernel):
    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              sigma: torch.Tensor) -> KernelMatrixObject:
        # todo is it a reason of strange sigma??
        # gamma = 1 / (2 * sigma ** 2)
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


class MaternKernelFunction(BaseKernel):
    def __init__(self,
                 nu: float,
                 length_scale: float = 1.0):
        self.gpy_kernel = MaternKernel(nu=nu, length_scale=length_scale)

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor) -> KernelMatrixObject:
        k_xy = self.gpy_kernel(x, y).evaluate()
        k_xx = self.gpy_kernel(x, x).evaluate()
        k_yy = self.gpy_kernel(y, y).evaluate()

        return KernelMatrixObject(k_xx=k_xx, k_yy=k_yy, k_xy=k_xy)
