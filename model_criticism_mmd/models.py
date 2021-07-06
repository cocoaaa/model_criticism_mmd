import numpy

from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch import BaseKernel

import collections
import dataclasses
import numpy as np
import typing
import nptyping
import os
import pickle
import torch


MmdValues = collections.namedtuple('MmdValues', ('mmd', 'ratio'))
TypeInputData = typing.Union[torch.Tensor, nptyping.NDArray[(typing.Any, typing.Any), typing.Any]]
TypeScaleVector = nptyping.NDArray[(typing.Any, typing.Any), typing.Any]


@dataclasses.dataclass
class TrainingLog(object):
    epoch: int
    avg_mmd_training: float
    avg_obj_train: float
    mmd_validation: float
    obj_validation: float
    sigma: typing.Optional[float]
    scales: nptyping.NDArray[(typing.Any,), typing.Any]


@dataclasses.dataclass
class TrainedMmdParameters(object):
    scales: nptyping.NDArray[(typing.Any, typing.Any), typing.Any]
    training_log: typing.List[TrainingLog]
    x_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
    y_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
    sigma: typing.Optional[np.float] = None
    kernel_function_obj: BaseKernel = None
    func_mapping_network: typing.Any = None

    def to_npz(self, path_npz: str):
        assert os.path.exists(os.path.dirname(path_npz))
        dict_obj = dataclasses.asdict(self)
        del dict_obj['mapping_network']
        np.savez(path_npz, **dict_obj)
        logger.info(f'saved as {path_npz}')

    def to_pickle(self, path_pickle: str):
        assert os.path.exists(os.path.dirname(path_pickle))
        f = open(path_pickle, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


class TrainerBase(object):
    def train(self, **kwargs) -> TrainedMmdParameters:
        raise NotImplementedError()

    def mmd_distance(self, **kwargs):
        raise NotImplementedError()


# todo save the data on storage, not on memory.
class TwoSampleDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 x: TypeInputData,
                 y: TypeInputData,
                 device_obj: torch.device,
                 value_padding: float = np.nan):
        assert isinstance(x, (torch.Tensor, numpy.ndarray))
        assert isinstance(y, (torch.Tensor, numpy.ndarray))
        self.x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y)
        self.length_x = len(x)
        self.length_y = len(y)
        self.value_padding = value_padding
        self.device_obj = device_obj
        if self.length_x != self.length_y:
            logger.warning(f'x and y has different sample size. '
                           f'I do not guarantee correct behaviors of training-process.')
        else:
            logger.debug(f'input data N(sample-size)={x.shape[0]}, N(dimension)={y.shape[1]}')
        # end if
        assert x.shape[-1] == y.shape[-1]
        self.size_dimension = x.shape[-1]

    def get_dimension(self):
        return self.size_dimension

    def get_all_item(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y

    def __getitem__(self, index):
        if index >= self.length_x:
            empty_array = np.empty(self.y[index].shape)
            empty_array[:] = self.value_padding
            return torch.tensor(empty_array), self.y[index]
        elif index >= self.length_y:
            empty_array = np.empty(self.x[index].shape)
            empty_array[:] = self.value_padding
            return self.x[index], torch.tensor(empty_array)
        elif index > self.length_x and self.length_y:
            raise Exception()
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        if self.length_x != self.length_y:
            return max([self.length_x, self.length_y])
        else:
            return self.length_x
