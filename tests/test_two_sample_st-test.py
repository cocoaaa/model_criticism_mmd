from model_criticism_mmd.supports import distribution_generator
from model_criticism_mmd import ModelTrainerTorchBackend, MMD, split_data
from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction
from model_criticism_mmd.supports import mmd_two_sample_test
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.models import TwoSampleDataSet
import numpy as np
import torch


def test_two_sample_sttest():
    n_train = 1500
    n_test = 500
    num_epochs = 100
    np.random.seed(np.random.randint(2 ** 31))
    x_train, y_train, x_test, y_test = distribution_generator.generate_data(n_train=n_train, n_test=n_test)

    trainer = ModelTrainerTorchBackend(mmd_estimator=MMD(kernel_function_obj=BasicRBFKernelFunction()))
    dataset_train = TwoSampleDataSet(device_obj=torch.device('cpu'), x=x_train, y=y_train)
    dataset_val = TwoSampleDataSet(x=x_test, y=y_test, device_obj=torch.device('cpu'))

    trained_obj = trainer.train(dataset_training=dataset_train,
                                dataset_validation=dataset_val,
                                num_epochs=num_epochs,
                                batchsize=200,
                                opt_log=True)

    p_val, stat, samps = mmd_two_sample_test.rbf_mmd_test(x=x_test, y=y_test, trained_params=trained_obj,
                                                          bandwidth='trained')
    logger.info(f'p-val: {p_val}')


if __name__ == '__main__':
    test_two_sample_sttest()

