from model_criticism_mmd.models.datasets import TwoSampleDataSet
from model_criticism_mmd.backends.kernels_torch.base import BaseKernel
from model_criticism_mmd.backends.backend_torch import ModelTrainerTorchBackend, MMD
from model_criticism_mmd.models import TrainedMmdParameters
from model_criticism_mmd.logger_unit import logger


import typing
import torch
import dataclasses


@dataclasses.dataclass
class SelectedKernel(object):
    kernel: BaseKernel
    test_power: float
    trained_mmd_parameter: typing.Optional[TrainedMmdParameters] = None

    def __str__(self):
        return f'Kernel-type: {self.kernel}. Test-power: {self.test_power}'

    def __repr__(self):
        return self.__str__()


class SelectionKernels(object):
    def __init__(self,
                 candidate_kernels: typing.List[typing.Tuple[typing.Optional[torch.Tensor], BaseKernel]],
                 dataset_validation: TwoSampleDataSet,
                 device_obj: torch.device = torch.device('cpu'),
                 is_training: bool = False,
                 dataset_training: TwoSampleDataSet = None,
                 num_epochs: int = 500,
                 batchsize: int = 256):
        """A class to select the best Kernel for the given dataset. The class selects the best kernel by test-power
        which is represented "ratio" in the codes.

        Args:
            candidate_kernels: List object. Elements of lists are tuple (scale-vector, kernel object).
            The scale-vector can be None object. If the vector is None, the vector is initialized with -1 values.
            dataset_validation: a dataset object for validations.
            device_obj: device object of Torch.
            is_training: boolean. If True, then it runs optimization of parameters.
            The initial value is the given parameter.
            dataset_training: a dataset object which is required only when is_training = True.
            num_epochs: #epochs only when is_training = True.
            batchsize: batchsize only when is_training = True.
        """
        self.is_training = is_training
        self.candidate_kernels = candidate_kernels
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        self.device_obj = device_obj
        self.num_epochs = num_epochs
        self.batchsize = batchsize

    def run_selection(self, **kwargs) -> typing.List[SelectedKernel]:
        """Run a selection of kernels.

        Args:
            **kwargs: Arguments for training,
            except dataset_training, dataset_validation, num_epochs, batchsize, initial_scale

        Returns: list of SelectedKernel object.
        """
        results = []
        for scale, kernel_obj in self.candidate_kernels:
            if scale is None:
                size_vector_dimension, size_vector_short = self.dataset_validation.get_dimension()
                vector_one = torch.ones(size_vector_dimension)
                scale = vector_one
            # end if
            mmd_estimator = MMD(kernel_function_obj=kernel_obj, scales=scale, device_obj=self.device_obj)
            if self.is_training:
                trainer_obj = ModelTrainerTorchBackend(mmd_estimator=mmd_estimator, device_obj=self.device_obj)
                trained_params = trainer_obj.train(dataset_training=self.dataset_training,
                                                   dataset_validation=self.dataset_validation,
                                                   num_epochs=self.num_epochs,
                                                   batchsize=self.batchsize,
                                                   initial_scale=scale,
                                                   **kwargs)
                trained_mmd_estimator = MMD(kernel_function_obj=trained_params.kernel_function_obj,
                                            scales=torch.tensor(trained_params.scales),
                                            device_obj=self.device_obj)
                mmd_estimator = trained_mmd_estimator
            else:
                trained_params = None
            # end if
            val_x, val_y = self.dataset_validation.get_all_item()
            mmd_result = mmd_estimator.mmd_distance(x=val_x, y=val_y)
            test_power = mmd_result.ratio.detach().cpu().numpy()[0]
            logger.info(f'Kernel-type: {kernel_obj} Ratio: {test_power}')
            results.append(SelectedKernel(kernel_obj, test_power, trained_params))
        # end for
        sorted_result = list(sorted(results, key=lambda t: t.test_power, reverse=True))
        return sorted_result
