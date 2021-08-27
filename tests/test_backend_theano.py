import pathlib

import numpy as np
import os
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd import ModelTrainerTheanoBackend
from model_criticism_mmd.supports.distribution_generator import generate_data


def test_devel(resource_path_root: pathlib.Path):
    num_epochs = 100
    path_trained_model = './save_test.npz'

    np.random.seed(np.random.randint(2**31))
    array_obj = np.load(resource_path_root / 'eval_array.npz')
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']
    trainer = ModelTrainerTheanoBackend()

    init_scale = np.array([0.05, 0.55])
    trained_obj = trainer.train(x=x_train,
                                y=y_train,
                                num_epochs=num_epochs,
                                init_sigma_median=False,
                                opt_strategy='nesterov_momentum',
                                x_val=x_test,
                                y_val=y_test,
                                opt_log=True,
                                opt_sigma=True,
                                init_scales=init_scale,
                                init_log_sigma=0.0)
    trained_obj.to_pickle(path_trained_model)

    import math
    logger.info(f'exp(sigma)={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')

    mmd2, t_value = trainer.mmd_distance(x=x_test, y=y_test, sigma=trained_obj.sigma)
    logger.info(f'MMD^2 = {mmd2}')
    os.remove(path_trained_model)


def test_example_ard_kernel():
    n_train = 1500
    n_test = 500
    num_epochs = 100
    path_trained_model = './trained_mmd.pickle'

    np.random.seed(np.random.randint(2**31))
    x_train, y_train, x_test, y_test = generate_data(n_train=n_train, n_test=n_test)
    trainer = ModelTrainerTheanoBackend()
    trained_obj = trainer.train(x=x_train,
                                y=y_train,
                                num_epochs=num_epochs,
                                init_sigma_median=False,
                                opt_strategy='nesterov_momentum',
                                x_val=x_test,
                                y_val=y_test,
                                opt_log=True,
                                opt_sigma=True)
    trained_obj.to_pickle(path_trained_model)
    import math
    logger.info(f'sigma={math.exp(trained_obj.sigma)} scales={trained_obj.scales}')

    mmd2, t_value = trainer.mmd_distance(x=x_test, y=y_test, sigma=trained_obj.sigma)
    logger.info(f'MMD^2 = {mmd2}')


if __name__ == '__main__':
    test_devel(pathlib.Path('./resources'))
    test_example_ard_kernel()
