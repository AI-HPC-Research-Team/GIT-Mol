import os
import numpy as np
import random
import torch
import rdkit.Chem as Chem

import datetime
from utils.ditributed_utils import mean_reduce

import logging
logger = logging.getLogger(__name__)

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        if mol is not None:
            return True
        else:
            return False
    except:
        return False

def print_model_info(model, level=0, prefix=''):
    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        total_params_module = sum(p.numel() for p in module.parameters())
        trainable_params_module = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += total_params_module
        trainable_params += trainable_params_module

        print(f"{prefix}Module: {name} | Total parameters: {total_params_module} | Trainable parameters: {trainable_params_module}")

        if level > 0:
            print_model_info(module, level=level-1, prefix=prefix + '  ')

    if prefix == '':
        print(f"Total parameters: {total_params} | Trainable parameters: {trainable_params} | Trainable ratio: {trainable_params / total_params:.2%}")

class EarlyStopping(object):
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))
        else:
            dt = datetime.datetime.now()
            filename = filename+"_{}_{:02d}-{:02d}-{:02d}.pth".format(dt.date(), dt.hour, dt.minute, dt.second)
        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.scores = []
        self.avg_score = 0
        self.std_dev = 0

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            self.scores = []
        else:
            self.counter += 1
            self.scores.append(score)
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.avg_score = np.mean(self.scores)
                self.std_dev = np.std(self.scores)
                self.early_stop = True
        return self.early_stop, self.avg_score, self.std_dev, self.filename

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, distributed=False, local_rank=0, dest_device=0, world_size=1):
        self.reset()
        self.distributed = distributed
        self.local_rank = local_rank
        self.dest_device = dest_device
        self.world_size = world_size

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        if self.distributed:
            return mean_reduce(self.avg)

        return self.avg

def ToDevice(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple) or isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    elif isinstance(obj, str):
        pass
    else:
        return obj.to(device)