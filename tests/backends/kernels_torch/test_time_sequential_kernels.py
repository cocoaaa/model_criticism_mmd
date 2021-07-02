from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd import MMD, ModelTrainerTorchBackend
from pathlib import Path
import numpy as np
import torch


def test_soft_dtw_single(resource_path_root):
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = torch.tensor(array_obj['x'], requires_grad=True) + 100
    y_train = torch.tensor(array_obj['y']) - 100

    kernel_obj = kernels_torch.SoftDtwKernelFunctionTimeSample()
    k_matrix_obj = kernel_obj.compute_kernel_matrix(x_train, y_train)
    grad_outputs = torch.ones_like(k_matrix_obj.k_xx)
    grads = torch.autograd.grad(k_matrix_obj.k_xx, x_train, grad_outputs=grad_outputs)[0]
    # check if grad exists
    assert len(grads[grads > 0.0]) > 0


def test_soft_dtw_unit_time_sample(resource_path_root):
    len_x, len_y, dims = 400, 400, 5
    # input is (n-sample, n-time-series, n-features)
    x_train = torch.normal(15, 0.9, (100, len_x, dims))
    y_train = torch.normal(10, 0.5, (100, len_y, dims))

    x_val = torch.rand((5, 150, dims))
    y_val = torch.rand((5, 100, dims))
    device_obj = torch.device('cpu')
    kernel_function = kernels_torch.SoftDtwKernelFunctionTimeSample(gamma=0.1)
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)
    trained_obj = trainer.train(x_train, y_train, num_epochs=100, batchsize=5, x_val=x_val, y_val=y_val,
                                initial_scale=None)
    trainer.mmd_distance(x_val, y_val, is_detach=True)


if __name__ == '__main__':
    # test_soft_dtw_single(Path('../../resources'))
    test_soft_dtw_unit_time_sample(Path('../../resources'))
    # test_soft_dtw_unit_time_feature(Path('../../resources'))
