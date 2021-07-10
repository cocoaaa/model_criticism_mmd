from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd import MMD, ModelTrainerTorchBackend
from model_criticism_mmd.models import TwoSampleDataSet, TwoSampleIterDataSet
from pathlib import Path
import numpy as np
import torch


def test_soft_dtw_single():
    # x_sample, y_sample is for example observation-points.
    x_sample, y_sample, x_time_length, y_time_length = 10, 10, 250, 300
    # input is (n-sample, n-time-series, n-features)
    x_train = torch.normal(15, 0.9, (x_sample, x_time_length), requires_grad=True)
    y_train = torch.normal(10, 0.5, (y_sample, y_time_length))

    kernel_obj = kernels_torch.SoftDtwKernelFunctionTimeSample()
    k_matrix_obj = kernel_obj.compute_kernel_matrix(x_train, y_train)
    grad_outputs = torch.ones_like(k_matrix_obj.k_xx)
    grads = torch.autograd.grad(k_matrix_obj.k_xx, x_train, grad_outputs=grad_outputs)[0]
    # check if grad exists
    assert len(grads[grads > 0.0]) > 0


def test_soft_dtw_single_diff_length_sample():
    # x_sample, y_sample is for example observation-points.
    x_sample, y_sample, x_time_length, y_time_length = 10, 20, 250, 300
    # input is (n-sample, n-time-series, n-features)
    x_train = torch.normal(15, 0.9, (x_sample, x_time_length), requires_grad=True)
    y_train = torch.normal(10, 0.5, (y_sample, y_time_length))

    kernel_obj = kernels_torch.SoftDtwKernelFunctionTimeSample()
    k_matrix_obj = kernel_obj.compute_kernel_matrix(x_train, y_train)
    grad_outputs = torch.ones_like(k_matrix_obj.k_xx)
    grads = torch.autograd.grad(k_matrix_obj.k_xx, x_train, grad_outputs=grad_outputs)[0]
    # check if grad exists
    assert len(grads[grads > 0.0]) > 0


def test_soft_dtw_unit_time_sample():
    x_sample, y_sample, x_time_length, y_time_length = 5, 5, 150, 300
    # input is (n-sample, n-time-series, n-features)
    x_train = torch.normal(150, 5, (x_sample, x_time_length))
    y_train = torch.normal(150, 0.5, (y_sample, y_time_length))

    x_val = torch.normal(150, 50, (10, x_time_length))
    y_val = torch.normal(150, 0.5, (15, y_time_length))
    device_obj = torch.device('cpu')
    kernel_function = kernels_torch.SoftDtwKernelFunctionTimeSample(gamma=0.5,
                                                                    post_normalize=True,
                                                                    opt_sigma=False,
                                                                    max_value_post_normalization=1000)
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)
    dataset_train = TwoSampleIterDataSet(x=x_train, y=y_train, device_obj=device_obj)
    dataset_val = TwoSampleIterDataSet(x=x_val, y=y_val, device_obj=device_obj)
    trained_obj = trainer.train(dataset_training=dataset_train,
                                dataset_validation=dataset_val,
                                num_epochs=50,
                                batchsize=30,
                                initial_scale=None,
                                is_gc=False)
    trainer.mmd_distance(x_val, y_val, is_detach=True)


if __name__ == '__main__':
    #test_soft_dtw_single()
    #test_soft_dtw_single_diff_length_sample()
    test_soft_dtw_unit_time_sample()

