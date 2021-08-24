from model_criticism_mmd.backends.kernels_torch.matern_kernel import MaternKernelFunction
from model_criticism_mmd.backends.backend_torch import MMD, ModelTrainerTorchBackend, TrainedMmdParameters
from model_criticism_mmd.models.datasets import TwoSampleDataSet
import torch
import numpy as np
from pathlib import Path
import typing

np.random.seed(np.random.randint(2**31))


def data_processor(resource_path_root: Path
                   ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = torch.tensor(array_obj['x'])
    y_train = torch.tensor(array_obj['y'])
    x_test = torch.tensor(array_obj['x_test'])
    y_test = torch.tensor(array_obj['y_test'])

    return x_train, y_train, x_test, y_test


def test_matern_kernels(resource_path_root: Path):
    """System test of Matern kernel object.
    Test includes a case using the median estimation for lengthscale."""
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    x_train = torch.normal(mean=0.0, std=1.0, size=(100, 500))
    y_train = torch.normal(mean=0.0, std=1.0, size=(100, 500))

    for nu_parameter in (0.5, 1.5, 2.5):
        kernel_matern_median = MaternKernelFunction(nu=nu_parameter, device_obj=device_obj)
        kernel_matern_median.set_lengthscale(x_train, y_train)
        kernel_object_01 = kernel_matern_median.compute_kernel_matrix(x_train, y_train)
        assert torch.min(kernel_object_01.k_xx).item() >= 0.1
        assert torch.min(kernel_object_01.k_xy).item() >= 0.1
        assert torch.min(kernel_object_01.k_yy).item() >= 0.1

        kernel_matern_non_median = MaternKernelFunction(nu=nu_parameter, device_obj=device_obj)
        kernel_object_02 = kernel_matern_non_median.compute_kernel_matrix(x_train, y_train)
        assert torch.equal(kernel_object_01.k_xx, kernel_object_02.k_xx) is False
        assert torch.equal(kernel_object_01.k_xy, kernel_object_02.k_xy) is False
        assert torch.min(kernel_object_02.k_xx).item() <= 0.001
        assert torch.min(kernel_object_02.k_xy).item() <= 0.001
        assert torch.min(kernel_object_02.k_yy).item() <= 0.001


def test_matern_kernel_optimization(resource_path_root: Path):
    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x_train, y_train, x_test, y_test = data_processor(resource_path_root)
    ds_train = TwoSampleDataSet(x=x_train, y=y_train, device_obj=device_obj)
    ds_val = TwoSampleDataSet(x=x_test, y=y_test, device_obj=device_obj)
    for nu_value in (0.5, 1.5, 2.5):
        kernel_matern = MaternKernelFunction(nu=nu_value, lengthscale=-1.0, device_obj=device_obj)
        length_scale_initial = kernel_matern.lengthscale

        init_scale = torch.tensor(np.array([0.05, 0.55]))
        estimator = MMD(kernel_function_obj=kernel_matern, scales=init_scale, device_obj=device_obj)
        trainer = ModelTrainerTorchBackend(mmd_estimator=estimator, device_obj=device_obj)
        res_opt = trainer.train(dataset_training=ds_train, dataset_validation=ds_val, num_epochs=150)
        # check if a lengthscale is over-written automatically
        assert res_opt.kernel_function_obj.lengthscale != length_scale_initial
        # assert res_opt.training_log[-1].avg_obj_train < res_opt.training_log[0].avg_obj_train


def test_matern_kernel_time_series(resource_path_root: Path):
    N_DATA_SIZE = 500
    N_TIME_LENGTH = 100
    NOISE_MU_X = 0
    NOISE_SIGMA_X = 0.5
    NOISE_MU_Y = 0
    NOISE_SIGMA_Y = 0.5

    INITIAL_VALUE_AT_ONE = np.random.normal(0, 0.5, (N_DATA_SIZE,))

    device_obj = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    x_data_sample = np.zeros((N_DATA_SIZE, N_TIME_LENGTH))
    y_data_sample = np.zeros((N_DATA_SIZE, N_TIME_LENGTH))
    y_data_sample_laplase = np.zeros((N_DATA_SIZE, N_TIME_LENGTH))

    x_data_sample[:, 0] = INITIAL_VALUE_AT_ONE
    y_data_sample[:, 0] = INITIAL_VALUE_AT_ONE
    y_data_sample_laplase[:, 0] = INITIAL_VALUE_AT_ONE

    for time_t in range(0, N_TIME_LENGTH - 1):
        noise_x = np.random.normal(NOISE_MU_X, NOISE_SIGMA_X, (N_DATA_SIZE,))
        noise_y = np.random.normal(NOISE_MU_Y, NOISE_SIGMA_Y, (N_DATA_SIZE,))
        noise_y_laplase = np.random.laplace(NOISE_MU_Y, NOISE_SIGMA_Y, (N_DATA_SIZE,))
        x_data_sample[:, time_t + 1] = x_data_sample[:, time_t].flatten() + noise_x
        y_data_sample[:, time_t + 1] = y_data_sample[:, time_t].flatten() + noise_y
        y_data_sample_laplase[:, time_t + 1] = y_data_sample_laplase[:, time_t].flatten() + noise_y_laplase
    # end for
    assert y_data_sample_laplase.shape == (N_DATA_SIZE, N_TIME_LENGTH)
    assert np.array_equal(x_data_sample, y_data_sample) is False
    kernel_matern = MaternKernelFunction(nu=0.5, device_obj=device_obj)
    init_scale = torch.tensor(np.array([0.5] * N_TIME_LENGTH))

    estimator = MMD(kernel_function_obj=kernel_matern, scales=init_scale, device_obj=device_obj)
    trainer = ModelTrainerTorchBackend(mmd_estimator=estimator, device_obj=device_obj)

    # (n-sample, n-time-length)
    ratio_training = 0.8
    ind_training = int((N_DATA_SIZE - 1) * ratio_training)
    x_train = x_data_sample[:ind_training, :]
    x_val = x_data_sample[ind_training:, :]
    y_train = y_data_sample[:ind_training, :]
    y_val = y_data_sample[ind_training:, :]
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_val, y_val, device_obj)
    res_opt = trainer.train(dataset_training=dataset_train, dataset_validation=dataset_val, num_epochs=150)
    assert res_opt.training_log[-1].avg_obj_train < res_opt.training_log[0].avg_obj_train

    # (n-time-series, n-sample)
    ratio_training = 0.8
    ind_training = int((N_TIME_LENGTH - 1) * ratio_training)
    x_train = x_data_sample.transpose()[:ind_training, :]
    x_val = x_data_sample.transpose()[ind_training:, :]
    y_train = y_data_sample.transpose()[:ind_training, :]
    y_val = y_data_sample.transpose()[ind_training:, :]
    dataset_train = TwoSampleDataSet(x_train, y_train, device_obj)
    dataset_val = TwoSampleDataSet(x_val, y_val, device_obj)
    res_opt = trainer.train(dataset_training=dataset_train, dataset_validation=dataset_val, num_epochs=150)
    assert res_opt.training_log[-1].avg_obj_train < res_opt.training_log[0].avg_obj_train


if __name__ == '__main__':
    #test_matern_kernels(Path('../../resources'))
    test_matern_kernel_time_series(Path('../../resources'))
    #test_matern_kernel_optimization(Path('../../resources'))
