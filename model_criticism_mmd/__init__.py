from model_criticism_mmd.logger_unit import logger
from model_criticism_mmd.backends.backend_torch import ModelTrainerTorchBackend, MMD
try:
    from model_criticism_mmd.backends.backend_theano import ModelTrainerTheanoBackend
except Exception as e:
    logger.debug(f'we fail to import a theano-backend because of {e}. To suppress this message. set logger = INFO')
