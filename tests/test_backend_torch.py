import pathlib
import typing

import numpy
import numpy as np
from pathlib import Path
from tempfile import mkdtemp

from model_criticism_mmd.backends.backend_torch import ModelTrainerTorchBackend, MMD
from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.kernels_torch import BasicRBFKernelFunction, MaternKernelFunction
from model_criticism_mmd.models import TwoSampleDataSet

import torch

np.random.seed(np.random.randint(2**31))


def data_processor(resource_path_root: Path
                   ) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = array_obj['x']
    y_train = array_obj['y']
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']

    return x_train, y_train, x_test, y_test


def test_optimizers(resource_path_root: Path):
    """test case to check any optimization modules"""
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj)
    trainer = ModelTrainerTorchBackend(
        MMD(kernel_function_obj=kernel_function, device_obj=device_obj), device_obj=device_obj)
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)
    trained_obj = trainer.train(dataset_training=dataset_train,
                                dataset_validation=dataset_val,
                                num_epochs=100,
                                initial_scale=init_scale,
                                is_training_auto_stop=True,
                                auto_stop_epochs=5,
                                auto_stop_threshold=0.00001,
                                name_optimizer='Adam',
                                args_optimizer={})


def test_auto_stop(resource_path_root: Path):
    """test case to check auto stop"""
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)
    trainer = ModelTrainerTorchBackend(
        MMD(kernel_function_obj=kernel_function, device_obj=device_obj), device_obj=device_obj)
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)
    trained_obj = trainer.train(dataset_training=dataset_train,
                                dataset_validation=dataset_val,
                                num_epochs=100000,
                                initial_scale=init_scale,
                                opt_log=True,
                                is_training_auto_stop=True,
                                auto_stop_epochs=5,
                                auto_stop_threshold=0.00001)
    logger.info(f'scales={trained_obj.scales}')
    mmd_value_trained = trainer.mmd_distance(x_test, y_test, is_detach=True)
    model_from_param = ModelTrainerTorchBackend.model_from_trained(trained_obj, device_obj=device_obj)
    mmd_value_from_params = model_from_param.mmd_distance(x_test, y_test, is_detach=True)
    assert (mmd_value_trained.mmd - mmd_value_from_params.mmd) < 0.01, \
        f"{mmd_value_trained.mmd}, {mmd_value_from_params.mmd}"
    logger.info(trained_obj.scales)


def test_l1_regularization(resource_path_root: Path):
    """Putting L1 regularization for objective value."""
    num_epochs = 100
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    dataset_train = TwoSampleDataSet(x_train, y_train)
    dataset_val = TwoSampleDataSet(x_test, y_test)
    kernel_function = BasicRBFKernelFunction(log_sigma=0.0)
    trainer_01 = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function))
    # test-power would be similar when reg_lambda:0.0 == reg_lambda is None
    trained_obj_none = trainer_01.train(dataset_training=dataset_train,
                                        dataset_validation=dataset_val,
                                        num_epochs=num_epochs,
                                        initial_scale=init_scale)

    trainer_02 = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function))
    trained_obj_zero_lambda = trainer_02.train(dataset_training=dataset_train,
                                               dataset_validation=dataset_val,
                                               num_epochs=num_epochs,
                                               initial_scale=init_scale,
                                               reg_strategy='l1',
                                               reg_lambda=0.0)

    trainer_03 = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function))
    trained_obj_zero_lambda = trainer_03.train(dataset_training=dataset_train,
                                               dataset_validation=dataset_val,
                                               num_epochs=num_epochs,
                                               initial_scale=init_scale,
                                               reg_strategy='l1',
                                               reg_lambda=0.1)


def test_non_negative_scales(resource_path_root: Path):
    num_epochs = 100
    path_trained_model = './trained_mmd_non_negative.pickle'
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)
    for kernel_function in [MaternKernelFunction(nu=0.5, device_obj=device_obj),
                            BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)]:
        trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)
        trained_obj = trainer.train(dataset_training=dataset_train,
                                    dataset_validation=dataset_val,
                                    num_epochs=num_epochs,
                                    initial_scale=init_scale,
                                    opt_log=True,
                                    is_scales_non_negative=True)
        trained_obj.to_pickle(path_trained_model)
        logger.info(f'scales={trained_obj.scales}')
        mmd_value_trained = trainer.mmd_distance(x_test, y_test, is_detach=True)
        model_from_param = ModelTrainerTorchBackend.model_from_trained(trained_obj, device_obj=device_obj)
        mmd_value_from_params = model_from_param.mmd_distance(x_test, y_test, is_detach=True)
        assert (mmd_value_trained.mmd - mmd_value_from_params.mmd) < 0.01, \
            f"{mmd_value_trained.mmd}, {mmd_value_from_params.mmd}"
        logger.info(trained_obj.scales)
        assert numpy.all(trained_obj.scales >= 0)


def test_multi_workers(resource_path_root: Path):
    num_epochs = 100
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)
    for kernel_function in [MaternKernelFunction(nu=0.5, device_obj=device_obj),
                            BasicRBFKernelFunction(log_sigma=0.0, device_obj=device_obj, opt_sigma=True)]:
        trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)
        trained_obj_multi = trainer.train(dataset_training=dataset_train,
                                          dataset_validation=dataset_val,
                                          num_epochs=num_epochs,
                                          initial_scale=init_scale,
                                          opt_log=True,
                                          num_workers=4)
        trained_obj_single = trainer.train(dataset_training=dataset_train,
                                           dataset_validation=dataset_val,
                                           num_epochs=num_epochs,
                                           initial_scale=init_scale,
                                           opt_log=True,
                                           num_workers=1)
        assert abs(trained_obj_multi.training_log[-1].mmd_validation - trained_obj_single.training_log[-1].mmd_validation) < 1.0
        # possible that objective value has much difference.
        # assert abs(trained_obj_multi.training_log[-1].obj_validation - trained_obj_single.training_log[-1].obj_validation) < 1.0


def test_devel(resource_path_root: Path):
    """A test case for general usage."""
    num_epochs = 100
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    init_scale = torch.tensor(np.array([0.05, 0.55]))
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_test, y_test, device_obj)
    for kernel_function in [
        BasicRBFKernelFunction(device_obj=device_obj, opt_sigma=True),
        MaternKernelFunction(nu=0.5, device_obj=device_obj),
    ]:
        trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)
        trained_obj = trainer.train(dataset_training=dataset_train,
                                    dataset_validation=dataset_val,
                                    num_epochs=num_epochs,
                                    initial_scale=init_scale,
                                    opt_log=True)
        path_dir_temp = mkdtemp()
        path_file_pickle = Path(path_dir_temp).joinpath('trained.pickle')
        path_file_torch = Path(path_dir_temp).joinpath('model.torch')
        trained_obj.to_pickle(str(path_file_pickle))
        trained_obj.save_torch_model(str(path_file_torch))

        # resume with pickle
        mmd_value_trained = trainer.mmd_distance(x_test, y_test, is_detach=True)
        model_from_param = ModelTrainerTorchBackend.model_from_trained(trained_obj, device_obj=device_obj)
        mmd_value_from_params = model_from_param.mmd_distance(x_test, y_test, is_detach=True)
        assert (mmd_value_trained.mmd - mmd_value_from_params.mmd) < 0.01, \
            f"{mmd_value_trained.mmd}, {mmd_value_from_params.mmd}"

        # reuse the model file. Check the saved model works
        with open(path_file_torch, 'rb') as f:
            mmd_estimator = torch.load(f)
        # end with
        res = mmd_estimator.mmd_distance(x_test, y_test)
        assert mmd_value_trained.mmd - res.mmd < 0.01


if __name__ == "__main__":
    test_optimizers(pathlib.Path('./resources'))
    test_devel(pathlib.Path('./resources'))
    test_auto_stop(pathlib.Path('./resources'))
    test_non_negative_scales(pathlib.Path('./resources'))
    test_multi_workers(pathlib.Path('./resources'))
    test_l1_regularization(pathlib.Path('./resources'))
