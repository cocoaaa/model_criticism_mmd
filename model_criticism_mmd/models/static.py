import collections
import typing
import torch
import nptyping


MmdValues = collections.namedtuple('MmdValues', ('mmd', 'ratio'))
TypeInputData = typing.Union[torch.Tensor, nptyping.NDArray[(typing.Any, typing.Any), typing.Any]]
TypeScaleVector = nptyping.NDArray[(typing.Any, typing.Any), typing.Any]
DEFAULT_DEVICE = torch.device('cpu')