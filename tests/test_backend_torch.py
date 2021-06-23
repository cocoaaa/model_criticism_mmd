import pathlib

import numpy as np
import os
from pathlib import Path

from model_criticism_mmd.backends.backend_torch import ModelTrainerTorchBackend, MMD
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction, MaternKernelFunction

import torch


def test_devel(resource_path_root: Path):
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2**31))
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for kernel_function in [#MaternKernelFunction(nu=0.5, device_obj=device_obj),
                            BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)]:
        trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)
        trained_obj = trainer.train(x_train,
                                    y_train,
                                    num_epochs=num_epochs,
                                    x_val=x_test,
                                    y_val=y_test,
                                    initial_scale=init_scale,
                                    opt_log=False)
        trained_obj.to_pickle(path_trained_model)
        import math
        logger.info(f'scales={trained_obj.scales}')
        if isinstance(kernel_function, BasicRBFKernelFunction):
            assert 0.0 < math.exp(trained_obj.kernel_function_obj.get_params(False)['log_sigma']) < 1.5, \
            f'tuned-sigma={math.exp(trained_obj.kernel_function_obj.get_params(False)["log_sigma"])}'
            os.remove(path_trained_model)
        # end if
        mmd_value_trained = trainer.mmd_distance(x_test, y_test, is_detach=True)
        model_from_param = ModelTrainerTorchBackend.model_from_trained(trained_obj, device_obj=device_obj)
        mmd_value_from_params = model_from_param.mmd_distance(x_test, y_test, is_detach=True)
        assert (mmd_value_trained.mmd - mmd_value_from_params.mmd) < 0.01, \
            f"{mmd_value_trained.mmd}, {mmd_value_from_params.mmd}"
        logger.info(trained_obj.scales)


if __name__ == "__main__":
    test_devel(pathlib.Path('./resources'))
