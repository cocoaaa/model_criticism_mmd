import pathlib

import numpy as np
import os
from pathlib import Path

from model_criticism_mmd.backends.backend_torch import ModelTrainerTorchBackend
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch import RBFKernelFunction, MaternKernelFunction

import torch


def test_devel(resource_path_root: Path):
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2**31))
    array_obj = np.load(resource_path_root / 'eval_array.npz')
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for kernel_function in [MaternKernelFunction(nu=0.5, device_obj=device_obj),
                            RBFKernelFunction(device_obj=device_obj)]:
        trainer = ModelTrainerTorchBackend(kernel_function_obj=kernel_function, device_obj=device_obj)
        trained_obj = trainer.train(x_train,
                                    y_train,
                                    num_epochs=num_epochs,
                                    x_val=x_test,
                                    y_val=y_test,
                                    initial_scale=init_scale,
                                    opt_sigma=True,
                                    opt_log=True,
                                    init_sigma_median=False)
        trained_obj.to_pickle(path_trained_model)
        import math
        logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')
        if isinstance(kernel_function, RBFKernelFunction):
            assert 0.0 < math.exp(trained_obj.sigma) < 1.5
            os.remove(path_trained_model)
        # end if
        mmd_value_trained = trainer.mmd_distance(x_test, y_test)
        model_from_param = ModelTrainerTorchBackend.model_from_trained(trained_obj, device_obj=device_obj)
        mmd_value_from_params = model_from_param.mmd_distance(x_test, y_test)
        assert mmd_value_trained == mmd_value_from_params, f"{mmd_value_trained}, {mmd_value_from_params}"
        logger.info(trained_obj.scales, trainer.log_sigma)


if __name__ == "__main__":
    test_devel(pathlib.Path('./resources'))
