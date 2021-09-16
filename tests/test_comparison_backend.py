import numpy
import pathlib
import torch
from model_criticism_mmd import ModelTrainerTorchBackend, ModelTrainerTheanoBackend, MMD, kernels_torch
from model_criticism_mmd.models import TwoSampleDataSet
from model_criticism_mmd.models.static import DEFAULT_DEVICE
from collections import namedtuple

OptimizationResult = namedtuple('OptimizationResult', ('x', 'y', 'theano', 'torch'))


def test_comparison(resource_path_root: pathlib.Path):
    """A test that we confirm results are almost similar between Theano-backend and Torch-backend.

    We check if MMD value from both backends are similar values.
    Also, we check the index with high variances == the index with high weight of scales vector.
    """
    device_obj_torch = DEFAULT_DEVICE

    size = 100
    n_trial = 3
    n_epoch = 500
    batch_size = 200

    result_stacks_with_sigma = []
    result_stacks_without_sigma = []
    initial_scales = numpy.array([0.1, 0.1, 0.1])
    initial_sigma = 0.0

    for i_trial in range(0, n_trial):
        x_1st_dim = numpy.random.normal(loc=1.0, scale=0.0, size=size)
        y_1st_dim = numpy.random.normal(loc=1.0, scale=50.0, size=size)

        x_2_and_3_dim = numpy.random.normal(loc=10.0, scale=0.2, size=(size, 2))
        y_2_and_3_dim = numpy.random.normal(loc=10.0, scale=0.2, size=(size, 2))

        x = numpy.concatenate([numpy.reshape(x_1st_dim, (size, 1)), x_2_and_3_dim], axis=1)
        y = numpy.concatenate([numpy.reshape(y_1st_dim, (size, 1)), y_2_and_3_dim], axis=1)

        x_train = x[:80]
        y_train = y[:80]
        x_val = x[80:]
        y_val = y[80:]

        dataset_train = TwoSampleDataSet(x_train, y_train, DEFAULT_DEVICE)
        dataset_val = TwoSampleDataSet(x_val, y_val, DEFAULT_DEVICE)

        # with sigma optimization
        trainer_theano = ModelTrainerTheanoBackend()
        trained_obj_theano = trainer_theano.train(x_train,
                                                  y_train,
                                                  num_epochs=n_epoch,
                                                  batchsize=batch_size,
                                                  opt_sigma=True,
                                                  x_val=x_val,
                                                  y_val=y_val,
                                                  init_sigma_median=False,
                                                  init_scales=initial_scales,
                                                  init_log_sigma=initial_sigma,
                                                  opt_log=True)

        mmd_estimator_sigma_opt = MMD(kernels_torch.BasicRBFKernelFunction(opt_sigma=False,
                                                                           device_obj=device_obj_torch,
                                                                           log_sigma=initial_sigma),
                                      device_obj=device_obj_torch,
                                      scales=torch.tensor(initial_scales))
        trainer_torch = ModelTrainerTorchBackend(mmd_estimator=mmd_estimator_sigma_opt,
                                                 device_obj=device_obj_torch)
        trained_obj_torch = trainer_torch.train(dataset_training=dataset_train,
                                                num_epochs=n_epoch,
                                                batchsize=batch_size,
                                                dataset_validation=dataset_val,
                                                initial_scale=torch.tensor(initial_scales),
                                                opt_log=True,
                                                is_use_lr_scheduler=False)
        result_stacks_with_sigma.append(OptimizationResult(x, y, trained_obj_theano, trained_obj_torch))

        # without sigma optimization
        trainer_theano = ModelTrainerTheanoBackend()
        trained_obj_theano = trainer_theano.train(x_train,
                                                  y_train,
                                                  num_epochs=n_epoch,
                                                  batchsize=batch_size,
                                                  opt_sigma=False,
                                                  x_val=x_val,
                                                  y_val=y_val,
                                                  init_sigma_median=False,
                                                  init_log_sigma=0.0,
                                                  init_scales=initial_scales)

        mmd_estimator = MMD(kernels_torch.BasicRBFKernelFunction(opt_sigma=False, device_obj=device_obj_torch),
                            device_obj=device_obj_torch)
        trainer_torch = ModelTrainerTorchBackend(mmd_estimator=mmd_estimator, device_obj=device_obj_torch)
        trained_obj_torch = trainer_torch.train(dataset_training=dataset_train,
                                                num_epochs=n_epoch,
                                                batchsize=batch_size,
                                                dataset_validation=dataset_val,
                                                is_use_lr_scheduler=False)
        result_stacks_without_sigma.append(OptimizationResult(x, y, trained_obj_theano, trained_obj_torch))
    # end for

    for set_with_sigma_opt, set_without_sigma_out in zip(result_stacks_with_sigma, result_stacks_without_sigma):
        # comparison of sigma_opt
        avg_mmd_training_torch = set_with_sigma_opt.torch.training_log[-1].mmd_validation
        avg_mmd_training_theano = set_with_sigma_opt.theano.training_log[-1].mmd_validation
        assert (avg_mmd_training_torch - avg_mmd_training_theano) < 1.0, \
            f'Result has significant difference! theano={avg_mmd_training_torch} torch={avg_mmd_training_torch}'

        avg_obj_train_torch = set_with_sigma_opt.torch.training_log[-1].obj_validation
        avg_obj_train_theano = set_with_sigma_opt.theano.training_log[-1].obj_validation
        assert (avg_obj_train_torch - avg_obj_train_theano) < 1.0, \
            f'Result has significant difference! theano={avg_obj_train_theano} torch={avg_obj_train_torch}'

        # comparison of sigma_opt = False
        avg_mmd_training_torch = set_without_sigma_out.torch.training_log[-1].mmd_validation
        avg_mmd_training_theano = set_without_sigma_out.theano.training_log[-1].mmd_validation
        assert (avg_mmd_training_torch - avg_mmd_training_theano) < 1.0, \
            f'Result has significant difference! theano={avg_mmd_training_torch} torch={avg_mmd_training_torch}'

        avg_obj_train_torch = set_without_sigma_out.torch.training_log[-1].obj_validation
        avg_obj_train_theano = set_without_sigma_out.theano.training_log[-1].obj_validation
        assert (avg_obj_train_torch - avg_obj_train_theano) < 1.0, \
            f'Result has significant difference! theano={avg_obj_train_theano} torch={avg_obj_train_torch}'

        # todo should check correspondence of index.
        x = set_with_sigma_opt.x
        y = set_with_sigma_opt.y

        dim_most_diff_variance: int = sorted(
            [(n_dim, abs(x[:, n_dim].var() - y[:, n_dim].var())) for n_dim in [0, 1, 2]],
            key=lambda t: t[1], reverse=True)[0][0]
        # end for

        assert int(numpy.argmax(set_without_sigma_out.torch.scales)) == dim_most_diff_variance, \
            f'{int(numpy.argmax(set_without_sigma_out.theano.scales))} != {dim_most_diff_variance}'


if __name__ == '__main__':
    test_comparison(pathlib.Path('./resources'))
