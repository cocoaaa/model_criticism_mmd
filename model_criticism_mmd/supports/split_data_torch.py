from model_criticism_mmd.models import TypeInputData, TwoSampleDataSet
import numpy as np
import typing
import torch


def to_tensor(data: np.ndarray) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError()


def split_data(x: TypeInputData,
               y: TypeInputData,
               device_obj: torch.device,
               ratio_train: float = 0.8) -> typing.Tuple[TwoSampleDataSet, TwoSampleDataSet]:
    # data conversion
    x__ = to_tensor(x)
    y__ = to_tensor(y)

    __split_index = int(len(x) * ratio_train)
    x_train__ = x__[:__split_index]
    x_val__ = x__[__split_index:]
    y_train__ = y__[:__split_index]
    y_val__ = y__[__split_index:]

    x_train__d = x_train__.to(device_obj)
    y_train__d = y_train__.to(device_obj)
    x_val__d = x_val__.to(device_obj)
    y_val__d = y_val__.to(device_obj)

    training_dataset = TwoSampleDataSet(device_obj=device_obj, x=x_train__d, y=y_train__d)
    val_dataset = TwoSampleDataSet(device_obj=device_obj, x=x_val__d, y=y_val__d)
    return training_dataset, val_dataset
