from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd import MMD, ModelTrainerTorchBackend
from scipy.spatial import distance
from pathlib import Path
import numpy as np


def test_any_distance_rbf_kernel_single(resource_path_root):
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = array_obj['x']
    y_train = array_obj['y']

    kernel_obj = kernels_torch.AnyDistanceRBFKernelFunction(distance.euclidean)
    k_matrix_obj = kernel_obj.compute_kernel_matrix(x_train, y_train)


def test_any_distance_rbf_kernel_unit(resource_path_root):
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = array_obj['x']
    y_train = array_obj['y']

    kernel_obj = kernels_torch.AnyDistanceRBFKernelFunction(distance.euclidean)
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_obj))
    trained_obj = trainer.train(x_train, y_train, num_epochs=50)


if __name__ == '__main__':
    test_any_distance_rbf_kernel_unit(Path('../../resources').absolute())
    test_any_distance_rbf_kernel_single(Path('../../resources').absolute())


