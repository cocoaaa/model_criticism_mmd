import typing
import torch
from typing import Union

import numpy as np
import numba as nb

from sklearn.metrics.pairwise import euclidean_distances
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel, KernelMatrixObject


FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')


class BasicRBFKernelFunction(BaseKernel):
    """A Kernel class with Euclidean-distance
    """
    def __init__(self,
                 device_obj: torch.device = device_default,
                 log_sigma: Union[float, torch.Tensor] = 0.0,
                 opt_sigma: bool = False):
        """
        Args:
            device_obj: torch.device object.
            log_sigma: sigma parameter of RBF kernel
            opt_sigma: if True, then sigma parameter will be optimized during training. False, not.
        """
        super().__init__(device_obj=device_obj)
        self.device_obj = device_obj
        self.opt_sigma = opt_sigma
        self.log_sigma = torch.tensor([log_sigma], requires_grad=opt_sigma, device=device_obj)

    @classmethod
    def init_with_median(cls,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         scales: torch.Tensor,
                         batchsize: int = 1000,
                         opt_sigma: bool = False,
                         device_obj: torch.device = torch.device('cpu')) -> "BasicRBFKernelFunction":
        """initialize RBFKernel with median heuristic.
        About the median heuristic, see https://arxiv.org/pdf/1707.07269.pdf

        Returns:
            RBFKernelFunction
        """
        # initialization of initial-sigma value
        logger.info("Getting median initial sigma value...")
        n_samp = min(500, x.shape[0], y.shape[0])

        samp = torch.cat([
            x[np.random.choice(x.shape[0], n_samp, replace=False)],
            y[np.random.choice(y.shape[0], n_samp, replace=False)],
        ])

        data_loader = torch.utils.data.DataLoader(samp, batch_size=batchsize, shuffle=False)
        reps = torch.cat([torch.mul(batch, scales) for batch in data_loader])
        np_reps = reps.detach().cpu().numpy()
        d2 = euclidean_distances(np_reps, squared=True)
        med_sqdist = np.median(d2[np.triu_indices_from(d2, k=1)])
        __init_log_simga = np.log(med_sqdist / np.sqrt(2)) / 2
        del samp, reps, d2, med_sqdist
        logger.info("initial sigma by median-heuristics {:.3g}".format(np.exp(__init_log_simga)))
        return cls(
            device_obj=device_obj,
            log_sigma=__init_log_simga,
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


class AnyDistanceRBFKernelFunction(BasicRBFKernelFunction):
    """You can use any distance metrics as your preference.
    The distance metric must satisfy distance = distance_metric(x, y)"""

    def __init__(self,
                 func_distance_metric: typing.Callable[[np.ndarray, np.ndarray], float],
                 device_obj: torch.device = device_default,
                 log_sigma: Union[float, torch.Tensor] = 0.0,
                 opt_sigma: bool = False):
        """
        Args:
            func_distance_metric: a distance metric that satisfies distance = f(x, y).
            device_obj: torch.device object.
            log_sigma: sigma parameter of RBF kernel
            opt_sigma: if True, then sigma parameter will be optimized during training. False, not.
        """
        super().__init__(device_obj=device_obj)
        self.func_distance_metric = func_distance_metric
        self.device_obj = device_obj
        self.opt_sigma = opt_sigma
        self.log_sigma = torch.tensor([log_sigma], requires_grad=opt_sigma, device=device_obj)

    @nb.jit
    def func_distance(self, x, y) -> float:
        d = self.func_distance_metric(x, y)
        return d

    @nb.jit
    def __compute_kernel_matrix(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        k_matrix = torch.empty(x_1.shape[0], x_2.shape[0])
        for i, x_sample in enumerate(x_1):
            for j, y_sample in enumerate(x_2):
                v_elem = self.func_distance(x_sample, y_sample)
                k_matrix[i, j] = v_elem
            # end for
        # end for
        return k_matrix

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, log_sigma: torch.Tensor = None,
                              **kwargs) -> KernelMatrixObject:
        if log_sigma is None:
            log_sigma = self.log_sigma
        # end if
        sigma = torch.exp(log_sigma)
        gamma = torch.div(1, (2 * torch.pow(sigma, 2)))
        k_xx = torch.exp(-1 * gamma * self.__compute_kernel_matrix(x, x))
        k_xy = torch.exp(-1 * gamma * self.__compute_kernel_matrix(x, y))
        k_yy = torch.exp(-1 * gamma * self.__compute_kernel_matrix(y, y))
        return KernelMatrixObject(k_xx=k_xx, k_yy=k_yy, k_xy=k_xy)
