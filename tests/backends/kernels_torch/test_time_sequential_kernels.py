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
    device_obj = torch.device('cpu')
    kernel_function = kernels_torch.SoftDtwKernelFunctionTimeSample()
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_function, device_obj=device_obj),
                                           device_obj=device_obj)

    len_x, len_y, dims = 20, 20, 5
    x = torch.rand((len_x, dims))
    y = torch.rand((len_y, dims))
    trained_obj = trainer.train(x, y, num_epochs=100)



if __name__ == '__main__':
    # test_soft_dtw_single(Path('../../resources'))
    test_soft_dtw_unit(Path('../../resources'))
