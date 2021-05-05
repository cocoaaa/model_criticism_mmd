from model_criticism_mmd.logger_unit import logger
import dataclasses
import numpy as np
import typing
import nptyping
import os
import pickle


@dataclasses.dataclass
class TrainingLog(object):
    epoch: int
    avg_mmd_training: float
    avg_obj_train: float
    mmd_validation: float
    obj_validation: float
    sigma: float
    scales: nptyping.NDArray[(typing.Any,), typing.Any]


@dataclasses.dataclass
class TrainedMmdParameters(object):
    sigma: np.float
    scales: nptyping.NDArray[(typing.Any, typing.Any), typing.Any]
    training_log: typing.List[TrainingLog]
    x_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
    y_train: typing.Optional[nptyping.NDArray[(typing.Any, typing.Any), typing.Any]] = None
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