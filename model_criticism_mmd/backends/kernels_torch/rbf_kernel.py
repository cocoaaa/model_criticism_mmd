import typing
import torch
from typing import Union

import numpy as np

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

    @staticmethod
    def get_median(rep_x: torch.Tensor,
                   rep_y: torch.Tensor,
                   minimum_sample: int = 500) -> float:
        logger.info("Getting median initial sigma value...")
        n_samp = min(minimum_sample, rep_x.shape[0], rep_y.shape[0])
        samp = torch.cat([
            rep_x[np.random.choice(rep_x.shape[0], n_samp, replace=False)],
            rep_y[np.random.choice(rep_y.shape[0], n_samp, replace=False)],
        ])
        np_reps = samp.detach().cpu().numpy()
        d2 = euclidean_distances(np_reps, squared=True)
        med_sqdist = np.median(d2[np.triu_indices_from(d2, k=1)])
        __init_log_sigma = np.log(med_sqdist / np.sqrt(2)) / 2
        del samp, d2, med_sqdist
        logger.info("initial sigma by median-heuristics {:.3g}".format(np.exp(__init_log_sigma)))
        return __init_log_sigma

    @classmethod
    def init_with_median(cls,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         scales: torch.Tensor,
                         opt_sigma: bool = False,
                         device_obj: torch.device = torch.device('cpu')) -> "BasicRBFKernelFunction":
        """initialize RBFKernel with median heuristic.
        About the median heuristic, see https://arxiv.org/pdf/1707.07269.pdf

        Returns:
            RBFKernelFunction
        """
        # initialization of initial-sigma value
        rep_x = torch.mul(x, scales)
        rep_y = torch.mul(y, scales)
        __init_log_sigma = cls.get_median(rep_x=rep_x, rep_y=rep_y)
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
