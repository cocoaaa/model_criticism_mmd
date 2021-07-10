import h5py
import tempfile
import pathlib
import torch
import typing
import numpy as np
from model_criticism_mmd.models.static import TypeInputData
from model_criticism_mmd.logger_unit import logger


# todo save the data on storage, not on memory.
class TwoSampleDataSet(torch.utils.data.Dataset):
    def __init__(self,
                 x: TypeInputData,
                 y: TypeInputData,
                 device_obj: torch.device,
                 value_padding: float = np.nan):
        assert isinstance(x, (torch.Tensor, np.ndarray))
        assert isinstance(y, (torch.Tensor, np.ndarray))
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

        if x.shape[-1] == y.shape[-1]:
            self.size_dimension = x.shape[-1]
            self.size_dimension_short = None
        else:
            logger.warning('The dimension size is different between x and y. '
                           'Possible in some cases. You can ignore this message '
                           'if the difference is as your intention.')
            self.size_dimension = max(x.shape[-1], y.shape[-1])
            self.size_dimension_short = min(x.shape[-1], y.shape[-1])

    def get_dimension_x(self) -> int:
        return self.x[-1]

    def get_dimension_y(self):
        return self.y[-1]

    def get_dimension(self) -> typing.Tuple[int, typing.Optional[int]]:
        return self.size_dimension, self.size_dimension_short

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


class TwoSampleIterDataSet(torch.utils.data.IterableDataset):
    def __init__(self,
                 x: TypeInputData,
                 y: TypeInputData,
                 device_obj: torch.device,
                 value_padding: float = np.nan,
                 working_dir: pathlib.Path = None):
        assert isinstance(x, (torch.Tensor, np.ndarray))
        assert isinstance(y, (torch.Tensor, np.ndarray))
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        y = y if isinstance(y, torch.Tensor) else torch.tensor(y)

        if working_dir is None:
            working_dir__ = pathlib.Path(tempfile.mktemp())
            working_dir__.mkdir()
        else:
            working_dir__ = working_dir
        # end if
        self.path_h5 = working_dir__.joinpath('data.h5')
        self.save_hdf5(x=x, y=y, path_destination=self.path_h5)

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

        if x.shape[-1] == y.shape[-1]:
            self.size_dimension = x.shape[-1]
            self.size_dimension_short = None
        else:
            logger.warning('The dimension size is different between x and y. '
                           'Possible in some cases. You can ignore this message '
                           'if the difference is as your intention.')
            self.size_dimension = max(x.shape[-1], y.shape[-1])
            self.size_dimension_short = min(x.shape[-1], y.shape[-1])

    @staticmethod
    def save_hdf5(x: torch.Tensor,
                  y: torch.Tensor,
                  path_destination: pathlib.Path) -> pathlib.Path:
        logger.info(f'Saving data into {path_destination}')
        f = h5py.File(path_destination, 'w', rdcc_nbytes=1024 ** 2 * 4000, rdcc_nslots=1e7)
        group = f.create_group("data")
        group.create_dataset("x", data=x, chunks=True)
        group.create_dataset("y", data=y, chunks=True)
        return path_destination

    @staticmethod
    def load_hdf5(path_file: pathlib.Path):
        logger.debug(f'Loading file from {path_file}')
        _hdf5 = h5py.File(path_file, 'r')
        _dataset = _hdf5['data']
        return _dataset

    @classmethod
    def from_disk(cls) -> "TwoSampleIterDataSet":
        raise Exception()

    def get_dimension_x(self) -> int:
        return self.x[-1]

    def get_dimension_y(self):
        return self.y[-1]

    def get_dimension(self) -> typing.Tuple[int, typing.Optional[int]]:
        return self.size_dimension, self.size_dimension_short

    def get_all_item(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y

    def __getitem__(self, index: int):
        # for management of __getitem__, we open the file here.
        # see the issue: https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983
        if not hasattr(self, 'x'):
            self.x = self.load_hdf5(self.path_h5)['x']
        # end if
        if not hasattr(self, 'y'):
            self.y = self.load_hdf5(self.path_h5)['y']
        # end if

        if index >= self.length_x:
            try:
                empty_array = np.empty(self.y[index].shape)
                empty_array[:] = self.value_padding
            except IndexError:
                print()
            return torch.tensor(empty_array), self.y[index]
        elif index >= self.length_y:
            empty_array = np.empty(self.x[index].shape)
            empty_array[:] = self.value_padding
            return self.x[index], torch.tensor(empty_array)
        elif index > self.length_x and self.length_y:
            raise Exception()
        else:
            return self.x[index], self.y[index]

    def __iter__(self):
        """called from DatasetLoader class."""
        for i in range(0, max([self.length_x, self.length_y])):
            yield self.__getitem__(i)
            i += 1
        return

    def __len__(self):
        if self.length_x != self.length_y:
            return max([self.length_x, self.length_y])
        else:
            return self.length_x
