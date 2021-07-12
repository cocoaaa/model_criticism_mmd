import typing

import numpy
import torch
import dataclasses
from typing import Union
from gpytorch.lazy import LazyEvaluatedKernelTensor


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
