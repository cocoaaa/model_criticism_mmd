import torch

from model_criticism_mmd import ModelTrainerTorchBackend, ModelTrainerTheanoBackend
import numpy


"""A test to confirm correctness of the algorithm.
The computed ARD weight has the following structure: 1st index highest, in contract 2nd and 3rd are low value.
"""


def test_case_ard_weight():
    """A test scenario. X and Y have 3 feature dimensions.
    Only the 1 dimension follows a distribution with wide variance. On the contract, the 2rd and 3rd dimension have similar values.
    Then, the trained ARD weight will be [high, low, low]
    """
    size = 100
    n_trial = 2
    n_epoch = 500
    batch_size = 200
    is_opt_sigma = False

    result_stacks = []
    init_scales = numpy.array([0.1, 0.1, 0.1])
    for i_trial in range(0, n_trial):
        x_1st_dim = numpy.random.normal(loc=1.0, scale=0.0, size=size)
        y_1st_dim = numpy.random.normal(loc=1.0, scale=50.0, size=size)

        x_2_and_3_dim = numpy.random.normal(loc=10.0, scale=0.2, size=(size, 2))
        y_2_and_3_dim = numpy.random.normal(loc=10.0, scale=0.2, size=(size, 2))

        x = numpy.concatenate([numpy.reshape(x_1st_dim, (size, 1)), x_2_and_3_dim], axis=1)
        y = numpy.concatenate([numpy.reshape(y_1st_dim, (size, 1)), y_2_and_3_dim], axis=1)

        for n_dim in [0, 1, 2]:
            print(f'{n_dim+1} dim. mean(x)={x[:,n_dim].mean()} mean(y)={y[:,n_dim].mean()} var(x)={x[:,n_dim].var()} var(y)={y[:,n_dim].var()}')
        # end for
        trainer_theano = ModelTrainerTheanoBackend()
        trained_obj_theano = trainer_theano.train(x, y,
                                                  num_epochs=n_epoch,
                                                  batchsize=batch_size,
                                                  opt_sigma=is_opt_sigma,
                                                  init_scales=init_scales)

        trainer_torch = ModelTrainerTorchBackend()
        trained_obj_torch = trainer_torch.train(x, y,
                                                num_epochs=n_epoch,
                                                batchsize=batch_size,
                                                opt_sigma=is_opt_sigma,
                                                initial_scale=torch.tensor(init_scales))

        result_stacks.append([
                                 (x, y),
                                 (trained_obj_theano.scales, trained_obj_torch.scales),
                                 (trained_obj_theano.sigma, trained_obj_torch.sigma)
        ])
    # end for
    for i_trial, trial_result in enumerate(result_stacks):
        weight_array_theano_backend = trial_result[1][0]
        weight_array_torch_backend = trial_result[1][1]
        assert weight_array_theano_backend[0] == max(weight_array_theano_backend)
        assert weight_array_torch_backend[0] == max(weight_array_torch_backend)
        print(weight_array_theano_backend, weight_array_torch_backend)
    # end for


if __name__ == '__main__':
    test_case_ard_weight()
