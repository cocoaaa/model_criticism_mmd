import typing

import numpy
import torch
import dataclasses
from typing import Union
from gpytorch.lazy import LazyEvaluatedKernelTensor
from model_criticism_mmd.logger_unit import logger
from sklearn.metrics.pairwise import euclidean_distances

FloatOrTensor = Union[float, torch.Tensor]
device_default = torch.device('cpu')


@dataclasses.dataclass
class KernelMatrixObject(object):
    k_xx: Union[torch.Tensor, LazyEvaluatedKernelTensor]
    k_yy: Union[torch.Tensor, LazyEvaluatedKernelTensor]
    k_xy: torch.Tensor


class BaseKernel(object):
    def __init__(self, device_obj: torch.device, possible_shapes: typing.Tuple[int, ...]):
        self.device_obj = device_obj
        self.possible_shapes = possible_shapes
        self.lengthscale = None

    def check_data_shape(self, data: torch.Tensor):
        if len(data.shape) not in self.possible_shapes:
            raise Exception(f'Input data has {len(data.shape)} tensor. '
                            f'But the kernel class expects {self.possible_shapes} tensor.')

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> KernelMatrixObject:
        raise NotImplementedError()

    def get_params(self, is_grad_param_only: bool = False) -> typing.Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def delete_padding_xy(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          padding_value: float = float('nan'),
                          target_dim: int = 1) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if torch.any(torch.isnan(x)):
            x__ = self.delete_padding(x, target_dim=target_dim, padding_value=padding_value)
        else:
            x__ = x
        # end if
        if torch.any(torch.isnan(y)):
            y__ = self.delete_padding(y, target_dim=target_dim, padding_value=padding_value)
        else:
            y__ = y
        # end if
        return x__, y__

    @staticmethod
    def delete_padding(tensor_obj: torch.Tensor,
                       padding_value: float = float('nan'),
                       target_dim: int = 1):
        """Delete a vector whose value is nan."""
        if numpy.isnan(padding_value):
            filtered_tensor = tensor_obj[~torch.any(tensor_obj.isnan(), dim=target_dim)]
        else:
            filtered_tensor = tensor_obj[~torch.any(tensor_obj == padding_value, dim=target_dim)]
        # end if
        return filtered_tensor

    @staticmethod
    def get_median(x: torch.Tensor,
                   y: torch.Tensor,
                   minimum_sample: int = 500,
                   is_log: bool = False) -> float:
        """Get a median value for kernel functions.
        The approach is shown in 'Large sample analysis of the median heuristic'

        Args:
            x: (samples, features)
            y: (samples, features)
            minimum_sample: a minimum value for sampling.
            is_log: If True, Eq. log(srqt(H_n) / 2) else sqrt(H_n / 2)

        Returns:
            computed median
        """
        logger.info("Getting median initial sigma value...")
        n_samp = min(minimum_sample, x.shape[0], y.shape[0])
        samp = torch.cat([
            x[numpy.random.choice(x.shape[0], n_samp, replace=False)],
            y[numpy.random.choice(y.shape[0], n_samp, replace=False)],
        ])
        np_reps = samp.detach().cpu().numpy()
        d2 = euclidean_distances(np_reps, squared=True)
        med_sqdist = numpy.median(d2[numpy.triu_indices_from(d2, k=1)])
        if is_log:
            res_value = numpy.log(med_sqdist / numpy.sqrt(2)) / 2

            del samp, d2, med_sqdist
        else:
            res_value = numpy.sqrt(med_sqdist / 2)
            del samp, d2, med_sqdist
        # end if
        logger.info("initial by median-heuristics {:.3g} with is_log={}".format(res_value, is_log))
        return res_value

    def set_lengthscale(self, x: torch.Tensor, y: torch.Tensor, is_log: bool = False) -> None:
        raise NotImplementedError()
