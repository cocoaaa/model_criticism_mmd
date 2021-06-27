from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd import MMD, ModelTrainerTorchBackend
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
import numpy as np
import torch


def print_graph(g, level=0):
    if g == None: return
    print('*'*level*4, g)
    for subg in g.next_functions:
        print_graph(subg[0], level+1)



def test_any_distance_rbf_kernel_single(resource_path_root):
    array_obj = np.load(str(resource_path_root / 'eval_array.npz'))
    x_train = torch.tensor(array_obj['x'])
    y_train = torch.tensor(array_obj['y'])
    x_test = array_obj['x_test']
    y_test = array_obj['y_test']

    kernel_obj_any = kernels_torch.AnyDistanceRBFKernelFunction(func_distance_metric=torch.cdist, opt_sigma=False)
    kernel_obj_hard = kernels_torch.BasicRBFKernelFunction(opt_sigma=False)
    k_obj_any = kernel_obj_any.compute_kernel_matrix(x_train, y_train)
    k_obj_hard = kernel_obj_hard.compute_kernel_matrix(x_train, y_train)

    ref_distance = euclidean_distances(x_train, x_train) ** 2
    sigma = np.exp(0.0)
    gamma = 1 / (2 * sigma ** 2)
    pure_numpy = np.exp(-1 * gamma * ref_distance)

    diff_kxx = k_obj_any.k_xx.cpu().detach().numpy() - pure_numpy
    diff_kxx2 = pure_numpy - k_obj_hard.k_xx.cpu().detach().numpy()
    diff_kxx3 = k_obj_any.k_xx.cpu().detach().numpy() - k_obj_hard.k_xx.cpu().detach().numpy()

    # todo should check it.
    #assert len(diff_kxx[diff_kxx > 0.01]) == 0
    #assert len(diff_kxx2[diff_kxx2 > 0.01]) == 0
    #assert len(diff_kxx3[diff_kxx3 > 0.01]) == 0
    # if distance function is able to handle matrix

    init_torch = [0.5, 0.5]
    kernel_obj = kernels_torch.AnyDistanceRBFKernelFunction(func_distance_metric=torch.cdist, opt_sigma=False)
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_obj))
    trained_obj = trainer.train(x_train, y_train, num_epochs=100, initial_scale=torch.tensor(init_torch),
                                x_val=x_test, y_val=y_test)
    #
    init_torch = [0.5, 0.5]
    kernel_obj = kernels_torch.BasicRBFKernelFunction(opt_sigma=False)
    trainer = ModelTrainerTorchBackend(MMD(kernel_function_obj=kernel_obj))
    trained_obj = trainer.train(x_train, y_train, num_epochs=100, initial_scale=torch.tensor(init_torch),
                                x_val=x_test, y_val=y_test)


if __name__ == '__main__':
    test_any_distance_rbf_kernel_single(Path('../../resources').absolute())


