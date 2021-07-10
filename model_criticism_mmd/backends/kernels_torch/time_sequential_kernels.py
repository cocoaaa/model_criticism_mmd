import typing
import torch
from typing import Union

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import KernelMatrixObject
from model_criticism_mmd.backends.kernels_torch.rbf_kernel import BasicRBFKernelFunction
from model_criticism_mmd.supports.metrics.soft_dtw import SoftDTW
from tslearn.metrics import soft_dtw

FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')


# todo wanna make this func speed up.
def func_compute_kernel_matrix_square(x: torch.Tensor,
                                      y: torch.Tensor,
                                      soft_dtw_generator) -> torch.Tensor:
    """Distance matrix only when x and y have the same number of data.
    Less computation thanks to a triangular matrix."""
    mat = torch.zeros((x.shape[0], y.shape[0]))
    for i_row in range(x.shape[0]):
        for i_col in range(i_row, y.shape[0]):
            x_sample = x[i_row]
            y_sample = y[i_col]
            x_tensor = x_sample.view(1, -1, 1)
            y_tensor = y_sample.view(1, -1, 1)
            __ = soft_dtw_generator.forward(x_tensor, y_tensor, is_return_matrix=False)
            mat[i_row, i_col] = __[0]
            # end if
        # end for
    # end for
    d_matrix = mat + mat.T - torch.diag(mat.diagonal())
    return d_matrix


def func_compute_kernel_matrix_generic(x: torch.Tensor,
                                       y: torch.Tensor,
                                       soft_dtw_generator) -> torch.Tensor:
    """Get a distance matrix"""
    mat = torch.zeros((x.shape[0], y.shape[0]))
    for i_row, x_sample in enumerate(x):
        for i_col, y_sample in enumerate(y):
            x_tensor = x_sample.view(1, -1, 1)
            y_tensor = y_sample.view(1, -1, 1)
            __ = soft_dtw_generator.forward(x_tensor, y_tensor, is_return_matrix=False)
            mat[i_row, i_col] = __[0]
            # end if
        # end for
    # end for
    return mat


class SoftDtwKernelFunctionTimeSample(BasicRBFKernelFunction):
    """A Kernel class when your data is temporal data.

    k(x, y) = RBF-kernel(x, y) where
    ||x-y|| of RBF-kernel is Soft-DTW, x, y \in R^n, 1d tensor.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 log_sigma: Union[float, torch.Tensor] = 100,
                 normalize: bool = False,
                 device_obj: torch.device = device_default,
                 opt_sigma: bool = False,
                 possible_shapes=(2,)):
        """

        Args:
            gamma: a parameter of SoftDTW
            log_sigma: a sigma of RBF kernel
            normalize: normalization of SoftDTW. normalization is valid only when x, y have the same number of samples.
            device_obj: device object of torch.
            opt_sigma: True or False.
        """
        self.gamma = gamma
        self.normalize = normalize
        self.log_sigma = torch.tensor([log_sigma]) if isinstance(log_sigma, float) else log_sigma
        self.opt_sigma = opt_sigma
        super().__init__(device_obj=device_obj, possible_shapes=possible_shapes, log_sigma=log_sigma)
        self.soft_dtw_generator = SoftDTW(use_cuda=True if device_obj.type == 'cuda' else False,
                                          gamma=gamma, normalize=normalize)

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        if 'log_sigma' not in kwargs:
            log_sigma = self.log_sigma
        else:
            log_sigma = kwargs['log_sigma']
        # end if

        d_xx = func_compute_kernel_matrix_square(x, x, soft_dtw_generator=self.soft_dtw_generator)
        d_yy = func_compute_kernel_matrix_square(y, y, soft_dtw_generator=self.soft_dtw_generator)
        if x.shape[0] == y.shape[0]:
            d_xy = func_compute_kernel_matrix_square(x, y, soft_dtw_generator=self.soft_dtw_generator)
        else:
            d_xy = func_compute_kernel_matrix_generic(x, y, soft_dtw_generator=self.soft_dtw_generator)
        # end if
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

