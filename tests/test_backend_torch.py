import pathlib
import random

import numpy as np
import os
from pathlib import Path

from model_criticism_mmd.backends.backend_torch import TypeInputData, ModelTrainerTorchBackend
from model_criticism_mmd.backends import backend_torch
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.supports import distribution_generator
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
    for kernel_function in [MaternKernelFunction(nu=0.5), RBFKernelFunction()]:
        trainer = ModelTrainerTorchBackend(kernel_function_obj=kernel_function)

        trained_obj = trainer.train(x_train,
                                    y_train,
                                    num_epochs=num_epochs,
                                    x_val=x_test,
                                    y_val=y_test,
                                    initial_scale=init_scale,
                                    opt_sigma=True, opt_log=True, init_sigma_median=False)
        trained_obj.to_pickle(path_trained_model)
        import math
        logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')
        if isinstance(kernel_function, RBFKernelFunction):
            assert np.linalg.norm(trained_obj.scales[0] - trained_obj.scales[1]) < 5
            assert 0.0 < math.exp(trained_obj.sigma) < 1.5
            os.remove(path_trained_model)
        # end if


def test_indifferent_sample_size(resource_path_root):
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2 ** 31))
    array_obj = np.load(resource_path_root / 'eval_array.npz')
    __x_train = array_obj['x']
    x_train = __x_train[random.sample(range(0, len(__x_train)-1), 300), :]
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']

    init_scale = torch.tensor(np.array([0.05, 0.55]))
    trainer = ModelTrainerTorchBackend()

    trained_obj = trainer.train(x_train,
                                y_train,
                                num_epochs=num_epochs,
                                x_val=x_test,
                                y_val=y_test,
                                initial_scale=init_scale,
                                opt_sigma=True, opt_log=True, init_sigma_median=False)
    trained_obj.to_pickle(path_trained_model)
    import math
    logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')
    os.remove(path_trained_model)


def test_example():
    n_train = 1500
    n_test = 500
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2**31))
    x_train, y_train, x_test, y_test = distribution_generator.generate_data(n_train=n_train, n_test=n_test)

    trainer = ModelTrainerTorchBackend()

    trained_obj = trainer.train(x_train,
                                y_train,
                                num_epochs=num_epochs,
                                x_val=x_test,
                                y_val=y_test,
                                batchsize=200,
                                opt_sigma=True, opt_log=True, init_sigma_median=False)
    trained_obj.to_pickle(path_trained_model)
    import math
    logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')
    mmd_on_test, ratio_on_test = trainer.mmd_distance(x_test, y_test)
    logger.info(f'MMD on test data = {mmd_on_test.detach().cpu().numpy()}')
    os.remove(path_trained_model)


def test_gpu_backend(resource_path_root):
    if not torch.cuda.is_available():
        # skip test
        pass
    else:
        num_epochs = 100
        path_trained_model = './trained_mmd.pickle'

        np.random.seed(np.random.randint(2 ** 31))
        array_obj = np.load(resource_path_root / 'eval_array.npz')
        __x_train = array_obj['x']
        x_train = __x_train[random.sample(range(0, len(__x_train)-1), 300), :]
        y_train = array_obj['y']
        x_test = array_obj['x_test']
        y_test = array_obj['y_test']

        init_scale = torch.tensor(np.array([0.05, 0.55]))
        for kernel_function in [RBFKernelFunction(), MaternKernelFunction(nu=0.5)]:
            trainer = ModelTrainerTorchBackend(kernel_function)

            trained_obj = trainer.train(x_train,
                                        y_train,
                                        num_epochs=num_epochs,
                                        x_val=x_test,
                                        y_val=y_test,
                                        initial_scale=init_scale,
                                        opt_sigma=True,
                                        opt_log=True,
                                        init_sigma_median=False,
                                        num_workers=0)
            trained_obj.to_pickle(path_trained_model)
            import math
            logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')
            os.remove(path_trained_model)
        # end for


if __name__ == "__main__":
    test_indifferent_sample_size(pathlib.Path('./resources'))
    test_devel(pathlib.Path('./resources'))
    test_gpu_backend(pathlib.Path('./resources'))
    test_example()
