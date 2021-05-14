import os
import sys
from pathlib import Path
import datetime
from tqdm import tqdm
from pprint import pprint
import time
import mlflow
import numpy as np
import warnings
import subprocess as sbp

from .py_logger import Logger
from .config import ArgParser


def get_datetime_str():
    cur_datetime = str(datetime.datetime.now())
    parts = cur_datetime.split()
    cur_datetime = '_'.join(parts)
    cur_datetime = cur_datetime.split('.')[0]
    return cur_datetime


def mkdir_safe(dir):
    dir = Path(dir)
    if not dir.is_dir():
        try:
            dir.mkdir(parents=True)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn("Failed creating path: {path}, probably a race"
                              " condition".format(path=str(dir)))


class ExpPath(object):
    """docstring for ExpPath"""

    def __init__(self, root_dir, dirs=['output', 'model_ckpts']):
        super(ExpPath, self).__init__()
        self.root_dir = Path(root_dir)
        self.time_s = get_datetime_str()
        self.log_file = self.root_dir / Path('logfile')
        self.param_file = self.root_dir / Path('params')
        self.dirs = [self.root_dir / Path(d) for d in dirs]

        self.paths = {
            'root': str(self.root_dir),
            'log_file': str(self.log_file),
            'param_file': str(self.param_file)
        }

        mkdir_safe(self.root_dir)

        for i, d in enumerate(self.dirs):
            mkdir_safe(d)
            self.paths[dirs[i]] = str(d)

    def get_paths(self):
        return self.paths


def getGitInfo(strict: bool=False):
    """Get version and diff information.

    Args:
        strict (bool, optional): If True (default) will raise an exception when
        there are modified or un-tracked python files in the repository.
    """

    from pkg_resources import resource_filename

    #
    # Change path to the repository
    #
    #cwd = os.getcwd()
    #os.chdir(os.path.abspath(resource_filename(__name__, '..')))
    ver_cmd = ['git', 'rev-list', '--full-history', '--all', '--abbrev-commit']
    p = sbp.Popen(ver_cmd, stdout=sbp.PIPE, stderr=sbp.PIPE)#, encoding="utf-8")
    ver, err = p.communicate()

    #
    # Create a diff file.
    #
    if strict:
        #
        # Raise an exception if there are modified or un-tracked files in the folder.
        #
        strict_cmd = "git status -u | egrep -v 'ipynb|notebooks\/' | egrep '\.py$'"
        o, e = sbp.Popen(strict_cmd, stdout=sbp.PIPE, stderr=sbp.PIPE, shell=True)\
            .communicate()
        o = o.decode("utf-8")
        if o != '':
            uncommited_files = "\n".join(o.split())
            err_msg = "The working directory contains uncommited files:\n{}\n"\
                .format(uncommited_files)
            err_msg += "Please commit or run `getGitInfo` in unstrict mode"
            raise Exception(err_msg)

        diff = None
    else:
        diff_cmd = 'git diff -- "*.py"'
        p = sbp.Popen(diff_cmd, stdout=sbp.PIPE, stderr=sbp.PIPE,
                      shell=True)#, encoding="utf-8")
        diff, err = p.communicate()

    #os.chdir(cwd)

    ver = ver.strip().split()
    version = "%04d_%s" % (len(ver), ver[0].decode("utf-8"))

    return version, diff


class Experiment(object):
    def __init__(self, tag='', exp_name=None, use_mlflow=True, comment="", mlflow_port=5000, strict_git=False):
        super().__init__()
        self.tag = tag
        self.from_dir = exp_name
        self.parser = ArgParser()
        self.args = self.parser.get_args()
        self.root_dir = self.args.result
        self.time_stamp = get_datetime_str()
        self.comment = comment
        if exp_name is not None:
            self.base_dir = Path(self.root_dir) / Path(exp_name) / Path(tag + self.time_stamp)
            self.mlflow_dir = Path(self.root_dir) / Path(exp_name)
        elif self.args.exp_name is not None:
            exp_name = self.args.exp_name
            self.base_dir = Path(self.root_dir) / Path(exp_name) / Path(tag + self.time_stamp)
            self.mlflow_dir = Path(self.root_dir) / Path(exp_name)
        else:
            self.base_dir = Path(self.root_dir) / Path(tag + self.time_stamp)
            self.mlflow_dir = self.base_dir

        self.paths = ExpPath(str(self.base_dir)).get_paths()
        self.logfile = str(self.base_dir / Path(tag + 'logfile.txt'))
        self.log_master = Logger(self.logfile)
        self.param_file = Path(self.base_dir) / Path(tag + 'config.yaml')

        logger = self.log_master.get_logger('init')
        logger.info('Initializing exepriment...')
        logger.info(f'Log information will be saved at: {self.logfile}')
        logger.info(f'Parameters saved at: {str(self.param_file)}')
        self.parser.save_args(str(self.param_file))
        self.name = Path(sys.argv[0]).stem

        # log command line
        command_line = "python "
        command_line += " ".join(sys.argv)
        command_line += "\n"
        with open(os.path.join(str(self.base_dir), 'cmdline.txt'), 'w') as f:
            f.write(command_line)

        # log git info
        try:
            git_version, diff = getGitInfo(strict=strict_git)
        except:
            git_version, diff = "no_gitinfo", b""
        if diff is not None:
            with open(os.path.join(str(self.base_dir), f'git_diff-{git_version}.txt'), 'wb') as f:
                f.write(diff)

        self.batch_per_epoch_train = None
        self.batch_per_epoch_eval = None

        self.use_mlflow = use_mlflow
        if use_mlflow:
            self.mlflow_server = self._mlflow_server_default(mlflow_port)

        if self.args.seed is not None:
            np.random.seed(self.args.seed)

    @staticmethod
    def _mlflow_server_default(mlflow_port=5000):
        return os.environ.get("MLFLOW_SERVER", 'http://localhost:' + str(mlflow_port))

    def start(self):
        if not self.use_mlflow:
            self.run()
            return
            
        mlflow.set_tracking_uri(self.mlflow_server)
        experiment_id = mlflow.set_experiment(str(self.mlflow_dir))
        with mlflow.start_run(experiment_id=experiment_id):
            #
            # Log the run parametres to mlflow.
            #
            mlflow.log_param("base_dir", str(self.base_dir))
            mlflow.log_param("comment", self.comment)

            for k, v in sorted(vars(self.args).items()):
                mlflow.log_param(k, v)

            self.run()

    def run(self):
        raise NotImplementedError()



class TrainExperiment(Experiment):

    def run(self):
        self.setup_model()
        logger = self.log_master.get_logger('train')
        logger.info('Loading data...')
        train_loader, val_loader = self.get_dataloader()
        logger.info('Start training...')
        for epoch in range(self.args.epoch):
            loss = self.train_epoch(train_loader, epoch)
            metric = self.val_epoch(val_loader, epoch)
            self.train_epoch_callback(epoch, metric, loss)


    def train_epoch(self, train_loader, epoch):
        logger = self.log_master.get_logger('train')
        logger.info(f'Epoch {epoch: 3}, training...')
        running_loss = 0.0
        steps = 0
        t1 = time.time()
        for batch, data in enumerate(tqdm(train_loader, leave=False, ncols=70, unit='b', total=self.batch_per_epoch_train)):
            loss = self.train_batch(data, steps)
            running_loss += loss
            steps += 1
        running_loss /= steps
        elapsed = (time.time() - t1) / 60
        logger.info(f'Epoch: {epoch:3}, time elapsed: {elapsed:3.1} mins, running loss: {running_loss: .5}')
        if self.batch_per_epoch_train is None:
            self.batch_per_epoch_train = steps
        return running_loss


    def val_epoch(self, val_loader, epoch):
        logger = self.log_master.get_logger('eval')
        logger.info(f'Epoch {epoch: 3}, evaluating...')
        t1 = time.time()
        pred_list = []
        truth_list = []
        steps = 0
        for batch, data in enumerate(tqdm(val_loader, leave=False, ncols=70, unit='b', total=self.batch_per_epoch_val)):
            pred, truth = self.val_batch(data, steps)
            pred_list += pred
            truth_list += truth
            steps += 1
        metric = self.evaluate(pred=pred_list, truth=truth_list)
        elapsed = (time.time() - t1) / 60
        if self.batch_per_epoch_val is None:
            self.batch_per_epoch_val = steps

        if len(pred_list) == 0:
            logger.info(f'Epoch: {epoch:3}, time elapsed: {elapsed:3.1} mins')
        else:
            logger.info(f'Epoch: {epoch:3}, time elapsed: {elapsed:3.1} mins, metric: {metric: .5}')
        return metric

    def setup_model(self):
        """
        build model, optimizer and restore checkpoint
        """
        raise NotImplementedError()

    def get_dataloader(self):
        raise NotImplementedError()

    def reset_loader(self, loader, **kwargs):
        """
        need to re-shuffle dataset each epoch?
        """
        return loader

    def train_batch(self, data, step):
        raise NotImplementedError

    def val_batch(self, data, step):
        raise NotImplementedError()

    def evaluate(self, *, pred: list, truth: list):
        raise NotImplementedError()
            
    def train_epoch_callback(self, epoch, metric, loss):
        """
        adjust learning rate, save model ckpt, etc...
        """
        pass
    

