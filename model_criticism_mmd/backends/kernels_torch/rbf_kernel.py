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
                 opt_sigma: bool = False,
                 possible_shapes: typing.Tuple[int, ...] = (2,)):
        """
        Args:
            device_obj: torch.device object.
            log_sigma: sigma parameter of RBF kernel
            opt_sigma: if True, then sigma parameter will be optimized during training. False, not.
        """
        super().__init__(device_obj=device_obj, possible_shapes=possible_shapes)
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

# ------------------------------------------------------------------------------------------------
# NOTE: the code below seems to have issues (or bugs).
# The learning process does not work at all.


# from gpytorch.kernels import RBFKernel
# from gpytorch.kernels.rbf_kernel import postprocess_rbf
# from gpytorch.kernels.kernel import Distance
# from typing import Any
#
#
# class CustomMetric(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, x1, x2) -> Any:
#         v = torch.sum(x1 + x2)
#         ctx.save_for_backward(v)
#         return v
#
#     @staticmethod
#     def backward(ctx, grad_outputs: Any) -> Any:
#         return grad_outputs / ctx.saved_tensors[0], grad_outputs / ctx.saved_tensors[0]
#
#
# class CustomDistance(Distance):
#     def __init__(self, postprocess_script=postprocess_rbf, **kwargs):
#         super().__init__()
#         self._postprocess = postprocess_script
#         self.metric = CustomMetric.apply
#         self.metric_parameters = kwargs
#
#     def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
#         # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
#         d_tensor = torch.zeros((x1.shape[0], x2.shape[0]))
#
#         for i_elem1, elem_1 in enumerate(x1):
#             for j_elem2, elem_2 in enumerate(x2):
#                 # v = torch.sum(elem_1 + elem_2)
#                 # v = wasserstein_distance(elem_1, elem_2)
#                 v = self.metric(elem_1, elem_2)
#                 d_tensor[i_elem1, j_elem2] = v
#         return self._postprocess(d_tensor) if postprocess else d_tensor
#
#         adjustment = x1.mean(-2, keepdim=True)
#         x1 = x1 - adjustment
#         x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
#
#         # Compute squared distance matrix using quadratic expansion
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x1_pad = torch.ones_like(x1_norm)
#         if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
#             x2_norm, x2_pad = x1_norm, x1_pad
#         else:
#             x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#             x2_pad = torch.ones_like(x2_norm)
#         x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
#         x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
#         res = x1_.matmul(x2_.transpose(-2, -1))
#
#         if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
#             res.diagonal(dim1=-2, dim2=-1).fill_(0)
#
#         # Zero out negative values
#         res.clamp_min_(0)
#         return self._postprocess(res) if postprocess else res
#
#     def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
#         # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
#         res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
#         res = res.clamp_min_(1e-30).sqrt_()
#         return self._postprocess(res) if postprocess else res
#
#
# class AnyDistanceRBFKernelFunction(BasicRBFKernelFunction):
#     """You can use any distance metrics as your preference.
#     The distance metric must satisfy distance = distance_metric(x, y)"""
#
#     def __init__(self,
#                  device_obj: torch.device = device_default,
#                  log_sigma: Union[float, torch.Tensor] = 0.0,
#                  opt_sigma: bool = False,
#                  **kwargs):
#         """
#         Args:
#             func_distance_metric: a distance metric that satisfies distance = f(x, y).
#             device_obj: torch.device object.
#             log_sigma: sigma parameter of RBF kernel
#             opt_sigma: if True, then sigma parameter will be optimized during training. False, not.
#         """
#         super().__init__(device_obj=device_obj)
#         self.device_obj = device_obj
#         self.opt_sigma = opt_sigma
#         self.log_sigma = torch.tensor([log_sigma], requires_grad=opt_sigma, device=device_obj)
#         self.rbf_kernel_gpy = RBFKernel()
#         self.rbf_kernel_gpy.distance_module = CustomDistance(**kwargs)
#
#     def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, log_sigma: torch.Tensor = None,
#                               **kwargs) -> KernelMatrixObject:
#         if log_sigma is None:
#             log_sigma = self.log_sigma
#         # end if
#         sigma = torch.exp(log_sigma)
#         gamma = torch.div(1, (2 * torch.pow(sigma, 2)))
#         is_matrix_wise = False
#
#         # todo need length scale off
#         k_xx = self.rbf_kernel_gpy.forward(x, x)
#         k_yy = self.rbf_kernel_gpy.forward(y, y)
#         k_xy = self.rbf_kernel_gpy.forward(x, y)
#         # k_xx = torch.exp(-1 * gamma * self.func_distance(x, x, torch.tensor([1.0]), self.func_distance_metric, is_matrix_wise) ** 2)
#         # k_xy = torch.exp(-1 * gamma * self.func_distance(x, y, torch.tensor([1.0]), self.func_distance_metric, is_matrix_wise) ** 2)
#         # k_yy = torch.exp(-1 * gamma * self.func_distance(y, y, torch.tensor([1.0]), self.func_distance_metric, is_matrix_wise) ** 2)
#         return KernelMatrixObject(k_xx=k_xx, k_yy=k_yy, k_xy=k_xy)
