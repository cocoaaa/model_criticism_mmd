import typing

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
