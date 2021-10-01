import pathlib
from logging import getLogger, FileHandler, DEBUG, INFO, Formatter
from model_criticism_mmd.models import TrainingLog
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
from tempfile import mkdtemp

try:
    import wandb
except ImportError:
    # raise ImportError('Install: pip install wandb.')
    pass
# end try


class BaseReport(object):
    def start(self, training_arguments: Dict[str, Any]):
        raise NotImplementedError()

    def record(self, log_object: TrainingLog) -> None:
        """"""
        raise NotImplementedError()

    def finish(self):
        raise NotImplementedError()


class OptimizerReport(BaseReport):
    pass


class OptimizerLogReport(OptimizerReport):
    def __init__(self,
                 path_log_file: str,
                 style: str = 'json'):
        super(OptimizerLogReport, self)
        assert Path(path_log_file).parent.exists()
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create file handler for logger.
        fh = FileHandler(path_log_file)
        fh.setLevel(level=INFO)
        fh.setFormatter(formatter)
        self.logger = getLogger(__name__)
        self.logger.addHandler(fh)
        self.logger.propagate = False
        self.style = style

    def start(self, training_arguments: Dict[str, Any]):
        self.logger.info(json.dumps(training_arguments))

    def record(self, log_object: TrainingLog) -> None:
        epoch_num = log_object.epoch
        obj_value_val = log_object.obj_validation
        mmd_val = log_object.mmd_validation
        ratio_val = log_object.ratio_validation
        if self.style == 'json':
            line = json.dumps({'epoch': epoch_num, 'obj_val': obj_value_val,
                               'mmd_val': mmd_val, 'ratio_val': ratio_val})
        else:
            line = f'[epoch {epoch_num}] on validation,'\
                   f'obj={obj_value_val} mmd={mmd_val} ratio={ratio_val}'
        # end if
        self.logger.info(line)

    def finish(self):
        self.logger.info('end optimization.')


class OptimizerWandbReport(OptimizerReport):
    def __init__(self,
                 project_name: str = 'model_criticism_mmd',
                 run_name: Optional[str] = None,
                 is_save_model: bool = True,
                 path_tmp_model_dir: Optional[pathlib.Path] = None,
                 model_name: str = 'mmd'):
        super(OptimizerWandbReport, self).__init__()
        if is_save_model and path_tmp_model_dir is None:
            self.path_tmp_model_dir = pathlib.Path(mkdtemp())
        elif is_save_model and path_tmp_model_dir is not None:
            self.path_tmp_model_dir = path_tmp_model_dir
        # end if

        self.run = wandb.init(project=project_name,
                              reinit=True)
        if run_name is None:
            wandb.run.name = f'build-{datetime.now()}'
        else:
            wandb.run.name = run_name
        # end if
        self.is_save_model = is_save_model
        self.model_name = model_name

    def start(self, training_arguments: Dict[str, Any]):
        self.run.config.update(training_arguments)

    def record(self, log_object: TrainingLog) -> None:
        epoch_num = log_object.epoch
        obj_value_val = log_object.obj_validation
        mmd_val = log_object.mmd_validation
        ratio_val = log_object.ratio_validation
        line_obj = {'epoch': epoch_num, 'obj_val': obj_value_val,
                    'mmd_val': mmd_val, 'ratio_val': ratio_val}
        self.run.log(line_obj)

    def finish(self):
        trained_model_artifact = wandb.Artifact(
            self.model_name,
            type="model",
            metadata=dict(wandb.config))
        trained_model_artifact.add_dir(self.path_tmp_model_dir)
        self.run.log_artifact(trained_model_artifact)
        self.run.finish()
