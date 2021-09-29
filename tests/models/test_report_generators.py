from model_criticism_mmd.models.report_generators import WandbReport, LogReport
from tempfile import mktemp
from model_criticism_mmd.models import TrainingLog
from model_criticism_mmd import ModelTrainerTorchBackend, MMD, TwoSampleDataSet, split_data
from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd.models.static import DEFAULT_DEVICE

import numpy as np
import os


def test_wandb():
    is_wandb_available = True if 'WANDB_API_KEY' in os.environ else False
    if is_wandb_available is False:
        # skip the test
        pass
    else:
        report_gen = WandbReport(project_name='test')
        dummy_np = np.zeros(10)
        test_log = TrainingLog(epoch=0, avg_mmd_training=0, avg_obj_train=0, mmd_validation=0, obj_validation=0,
                               scales=dummy_np, sigma=None)
        report_gen.record(test_log)
        report_gen.finish()


def test_log_report():
    path_temp_file = mktemp()
    dummy_np = np.zeros(10)
    test_log = TrainingLog(epoch=0, avg_mmd_training=0,
                           avg_obj_train=0, mmd_validation=0,
                           obj_validation=0, sigma=None, scales=dummy_np)
    log_reporter = LogReport(path_log_file=path_temp_file)
    log_reporter.record(test_log)
    log_reporter.finish()


def test_training_log_report():
    n_train = 400
    x = np.random.normal(3, 0.5, size=(500, 2))
    y = np.random.normal(3, 0.5, size=(500, 2))
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    path_temp_file = mktemp()
    log_reporter = LogReport(path_log_file=path_temp_file)

    dataset_train = TwoSampleDataSet(x_train, y_train)
    dataset_val = TwoSampleDataSet(x_test, y_test)

    mmd_estimator = MMD(kernels_torch.BasicRBFKernelFunction(opt_sigma=False))
    trainer = ModelTrainerTorchBackend(mmd_estimator=mmd_estimator)
    trained_obj = trainer.train(dataset_training=dataset_train,
                                dataset_validation=dataset_val,
                                num_epochs=10,
                                batchsize=200,
                                is_training_auto_stop=True,
                                auto_stop_epochs=10,
                                name_optimizer='Adam',  # class name of torch.optimizer
                                report_to=log_reporter)


def test_training_wandb_report():
    n_train = 400
    x = np.random.normal(3, 0.5, size=(500, 2))
    y = np.random.normal(3, 0.5, size=(500, 2))
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    is_wandb_available = True if 'WANDB_API_KEY' in os.environ else False
    if is_wandb_available is False:
        # skip the test
        return

    log_reporter = WandbReport()
    dataset_train = TwoSampleDataSet(x_train, y_train)
    dataset_val = TwoSampleDataSet(x_test, y_test)

    mmd_estimator = MMD(kernels_torch.BasicRBFKernelFunction(opt_sigma=False))
    trainer = ModelTrainerTorchBackend(mmd_estimator=mmd_estimator)
    trained_obj = trainer.train(dataset_training=dataset_train,
                                dataset_validation=dataset_val,
                                num_epochs=10,
                                batchsize=200,
                                is_training_auto_stop=True,
                                auto_stop_epochs=10,
                                name_optimizer='Adam',  # class name of torch.optimizer
                                report_to=log_reporter)


if __name__ == '__main__':
    test_training_log_report()
    test_training_wandb_report()



