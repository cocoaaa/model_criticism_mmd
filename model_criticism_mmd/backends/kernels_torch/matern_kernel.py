import torch
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel, KernelMatrixObject
from gpytorch.kernels.matern_kernel import MaternKernel

import typing

FloatOrTensor = typing.Union[float, torch.Tensor]
device_default = torch.device('cpu')


class MaternKernelFunction(BaseKernel):
    """A class for Matern Kernel."""
    def __init__(self,
                 nu: float,
                 device_obj: torch.device = device_default,
                 lengthscale: float = -1.0,
                 possible_shapes=(2,)):
        """init an object.
        Parameters of Matern Kernel is not for optimizations. It is supposed that nu and length_scale params are fixed.

        Args:
            nu: 0.5, 1.5, 2.5
            device_obj: torch.device
            lengthscale: lengthscale of Matern kernel. -1.0 represents "median heuristic"
        """
        super().__init__(device_obj=device_obj, possible_shapes=possible_shapes)
        self.nu = nu
        self.lengthscale = lengthscale
        if lengthscale == -1.0:
            self.gpy_kernel = MaternKernel(nu=nu, length_scale=lengthscale)
        else:
            self.gpy_kernel = MaternKernel(nu=nu)
        # end if
        if device_obj == torch.device('cuda'):
            self.gpy_kernel = self.gpy_kernel.cuda()
        # end if

    def set_lengthscale(self, x: torch.Tensor, y: torch.Tensor, is_log: bool = False) -> None:
        scale_value = self.get_median(x, y, is_log=is_log)
        self.lengthscale = scale_value
        self.gpy_kernel.lengthscale = scale_value

    def compute_kernel_matrix(self,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              **kwargs) -> KernelMatrixObject:
        k_xy = self.gpy_kernel(x, y).evaluate()
        k_xx = self.gpy_kernel(x, x).evaluate()
        k_yy = self.gpy_kernel(y, y).evaluate()

        return KernelMatrixObject(k_xx=k_xx, k_yy=k_yy, k_xy=k_xy)

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, FloatOrTensor]:
        if is_grad_param_only:
            return {}
        else:
            return {'nu': self.nu, 'lengthscale': self.lengthscale}
