from model_criticism_mmd.supports import distribution_generator
from model_criticism_mmd import ModelTrainerTorchBackend
from model_criticism_mmd.supports import mmd_two_sample_test
from model_criticism_mmd.logger_unit import logger
import numpy as np


def test_two_sample_sttest():
    n_train = 1500
    n_test = 500
    num_epochs = 100
    np.random.seed(np.random.randint(2 ** 31))
    x_train, y_train, x_test, y_test = distribution_generator.generate_data(n_train=n_train, n_test=n_test)

    trainer = ModelTrainerTorchBackend()

    trained_obj = trainer.train(x_train,
                                y_train,
                                num_epochs=num_epochs,
                                batchsize=200,
                                opt_sigma=True, opt_log=True, init_sigma_median=False)
    p_val, stat, samps = mmd_two_sample_test.rbf_mmd_test(x=x_test, y=y_test,
                                                          trained_params=trained_obj, bandwidth='trained')
    logger.info(f'p-val: {p_val}')

if __name__ == '__main__':
    test_two_sample_sttest()

