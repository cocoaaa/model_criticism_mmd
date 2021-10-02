import torch.nn

from model_criticism_mmd.backends.kernels_torch import BaseKernel

import dataclasses
import os
import pickle

from model_criticism_mmd.models.datasets import *
from model_criticism_mmd.models.static import *


@dataclasses.dataclass
class TrainingLog(object):
    epoch: int
    avg_mmd_training: float
    avg_obj_train: float
    mmd_validation: float
    obj_validation: float
    sigma: typing.Optional[float]
    scales: nptyping.NDArray[(typing.Any,), typing.Any]
    ratio_training: typing.Optional[float] = None
    ratio_validation: typing.Optional[float] = None


@dataclasses.dataclass
class TrainedMmdParameters(object):
    scales: nptyping.NDArray[(typing.Any, typing.Any), typing.Any]
    training_log: typing.List[TrainingLog]
    x_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
    y_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
    sigma: typing.Optional[np.float] = None
    kernel_function_obj: BaseKernel = None
    func_mapping_network: typing.Any = None
    ratio_validation: float = None
    torch_model: typing.Optional[torch.nn.Module] = None

    def __post_init__(self):
        self.ratio_validation = self.training_log[-1].ratio_validation

    def to_npz(self, path_npz: str) -> None:
        assert os.path.exists(os.path.dirname(path_npz))
        dict_obj = dataclasses.asdict(self)
        del dict_obj['mapping_network']
        np.savez(path_npz, **dict_obj)
        logger.info(f'saved as {path_npz}')

    def to_pickle(self, path_pickle: str) -> None:
        assert os.path.exists(os.path.dirname(path_pickle))
        f = open(path_pickle, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_torch_model(self, path_torch_model: str) -> None:
        assert os.path.exists(os.path.dirname(path_torch_model))
        assert self.torch_model is not None
        with open(path_torch_model, 'wb') as f:
            torch.save(self.torch_model, f)
        # end with
        logger.info(f'saved as {path_torch_model}')

class TrainerBase(object):
    def train(self, **kwargs) -> TrainedMmdParameters:
        raise NotImplementedError()

    def mmd_distance(self, **kwargs):
        raise NotImplementedError()

