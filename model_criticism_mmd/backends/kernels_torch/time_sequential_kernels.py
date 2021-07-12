import typing
import torch
from typing import Union

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import KernelMatrixObject
from model_criticism_mmd.backends.kernels_torch.rbf_kernel import BasicRBFKernelFunction
from model_criticism_mmd.supports.metrics.soft_dtw import SoftDTW


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
                 log_sigma: Union[float, torch.Tensor] = 1000,
                 post_normalize: bool = True,
                 max_value_post_normalization: int = 1000,
                 device_obj: torch.device = device_default,
                 opt_sigma: bool = False,
                 possible_shapes=(2,)):
        """

        Args:
            gamma: a parameter of SoftDTW
            log_sigma: a sigma of RBF kernel
            post_normalize: normalization of SoftDTW. A Distance matrix of SoftDTW is normalized into (0, max_value_post_normalization).
            max_value_post_normalization: only when post_normalize = True
            device_obj: device object of torch.
            opt_sigma: True or False.
        """
        self.gamma = gamma
        self.post_normalize = post_normalize
        self.log_sigma = torch.tensor([log_sigma]) if isinstance(log_sigma, float) else log_sigma
        self.opt_sigma = opt_sigma
        self.max_value_post_normalization = max_value_post_normalization
        super().__init__(device_obj=device_obj,
                         possible_shapes=possible_shapes, log_sigma=log_sigma,
                         opt_sigma=opt_sigma)
        self.soft_dtw_generator = SoftDTW(use_cuda=True if device_obj.type == 'cuda' else False,
                                          gamma=gamma, normalize=False)

    @staticmethod
    def __post_normalization(d_matrix: torch.Tensor, max_value: int):
        # moving into 0
        min_zero_d = d_matrix + (0 - torch.min(d_matrix))
        # assert torch.min(min_zero_d).detach().numpy().tolist() == 0.0
        denominator_ = max_value / torch.max(min_zero_d)
        post_d__ = min_zero_d * denominator_
        # assert abs(torch.max(post_d__).detach().numpy().tolist() - max_value) < 0.1, \
        #     f'{torch.max(post_d__).detach().numpy().tolist()} == {max_value}'
        return post_d__

    def post_normalization(self,
                           d_xx: torch.Tensor,
                           d_yy: torch.Tensor,
                           d_xy: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_xx__ = self.__post_normalization(d_xx, self.max_value_post_normalization)
        d_yy__ = self.__post_normalization(d_yy, self.max_value_post_normalization)
        d_xy__ = self.__post_normalization(d_xy, self.max_value_post_normalization)
        return d_xx__, d_yy__, d_xy__

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        if 'log_sigma' not in kwargs:
            log_sigma = self.log_sigma
        else:
            log_sigma = kwargs['log_sigma']
        # end if
        x__, y__ = self.delete_padding_xy(x, y, target_dim=1)

        d_xx = func_compute_kernel_matrix_square(x__, x__, soft_dtw_generator=self.soft_dtw_generator)
        d_yy = func_compute_kernel_matrix_square(y__, y__, soft_dtw_generator=self.soft_dtw_generator)
        if x.shape[0] == y.shape[0]:
            d_xy = func_compute_kernel_matrix_square(x__, y__, soft_dtw_generator=self.soft_dtw_generator)
        else:
            d_xy = func_compute_kernel_matrix_generic(x__, y__, soft_dtw_generator=self.soft_dtw_generator)
        # end if
        if self.post_normalize:
            d_xx_, d_yy_, d_xy_ = self.post_normalization(d_xx, d_yy, d_xy)
        else:
            d_xx_, d_yy_, d_xy_ = d_xx, d_yy, d_xy
        # end if
        gamma = torch.div(1, (2 * torch.pow(log_sigma, 2)))
        k_xx = torch.exp(-1 * gamma * torch.pow(d_xx_, 2))
        k_yy = torch.exp(-1 * gamma * torch.pow(d_yy_, 2))
        k_xy = torch.exp(-1 * gamma * torch.pow(d_xy_, 2))
        if torch.all(k_xy == 0):
            logger.warning('k_xy matrix has all 0 value. Check training condition.')
        # end if
        return KernelMatrixObject(k_xx, k_yy, k_xy)

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, torch.Tensor]:
        if is_grad_param_only:
            return {'log_sigma': self.log_sigma}
        else:
            return {'gamma': self.gamma, 'log_sigma': self.log_sigma,
                    'post_normalize': self.post_normalize,
                    'max_value_post_normalization': self.max_value_post_normalization}
