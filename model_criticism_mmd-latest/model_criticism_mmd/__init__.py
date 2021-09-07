from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.backend_torch import ModelTrainerTorchBackend, MMD, TwoSampleDataSet
from model_criticism_mmd.backends import kernels_torch
from model_criticism_mmd.supports.split_data_torch import split_data
from model_criticism_mmd.supports.permutation_tests import PermutationTest
from model_criticism_mmd.supports.selection_kernels import SelectionKernels
from model_criticism_mmd.supports.evaluate_stats_tests import TestResult, TestResultGroupsFormatter, StatsTestEvaluator
try:
    from model_criticism_mmd.backends.backend_theano import ModelTrainerTheanoBackend
except Exception as e:
    logger.debug(f'we fail to import a theano-backend because of {e}. To suppress this message. set logger = INFO')
