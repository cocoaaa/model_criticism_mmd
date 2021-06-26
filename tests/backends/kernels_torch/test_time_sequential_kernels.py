from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd import MMD, ModelTrainerTorchBackend
from pathlib import Path
import numpy as np
import torch


def test_soft_dtw_single(resource_path_root):
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = array_obj['x']
    y_train = array_obj['y']

    kernel_obj = kernels_torch.SoftDtwKernelFunctionTimeSample()
    k_matrix_obj = kernel_obj.compute_kernel_matrix(torch.tensor(x_train), torch.tensor(y_train))


def test_soft_dtw_unit(resource_path_root):
    len_x, len_y, dims = 350, 350, 5
    x_train = torch.normal(50, 150, (len_x, dims))
    y_train = torch.normal(10, 0.5, (len_y, dims))

    x_val = torch.rand((100, dims))
    y_val = torch.rand((100, dims))
    device_obj = torch.device('cpu')
    kernel_function = kernels_torch.SoftDtwKernelFunctionTimeSample(gamma=0.1, log_sigma=0.0)
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)

    trained_obj = trainer.train(x_train, y_train, num_epochs=100, batchsize=31, x_val=x_val, y_val=y_val,
                                initial_scale=None)



if __name__ == '__main__':
    # test_soft_dtw_single(Path('../../resources'))
    test_soft_dtw_unit(Path('../../resources'))
