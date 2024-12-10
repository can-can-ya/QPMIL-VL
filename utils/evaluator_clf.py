#######################################
# Evaluator for classification models
#######################################
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from sklearn import metrics
from sklearn.calibration import calibration_curve

from .tools import get_loss_function


class BinClf_Evaluator(object):
    """Performance evaluator for binary classification model"""
    def __init__(self, pos_label=1, **kws):
        super(BinClf_Evaluator, self).__init__()
        self.kws = kws
        self.pos_label = pos_label
        self.valid_functions = {
            'auc': self._auc,
            'acc': self._acc,
            'acc@mid': self._acc_mid_threshold,
            'acc_best': self._acc_best,
            'loss': self._loss,
            'recall': self._recall,
            'precision': self._precision,
            'f1_score': self._f1_score,
            'f1_score@mid': self._f1_score_mid_threshold,
            'ece': self._ece,
            'mce': self._mce
        }
        self.valid_metrics = ['auc', 'loss', 'acc', 'acc_best', 'acc@mid', 'recall', 'precision', 'f1_score', 
            'f1_score@mid', 'ece', 'mce']
        print("[setup] A binary classification evaluator is loaded.")

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics

    def ret_pred_and_gt(self):
        return self.y_hat, self.y

    def _pre_compute(self, data):
        """
        If data['y_hat'] is with a shape of 
          - [N, num_cls], then it will be recognized as a raw output (logit) from NN.
          - [N, 1] or [N, ], then it will be recognized as a prob output.
        """
        self.y, self.y_hat_full = data['y'], data['y_hat']

        if 'n' in data:
            self.num_samples = data['n']
        else:
            self.num_samples = None

        if len(self.y_hat_full.shape) == 2 and self.y_hat_full.shape[1] > 1:
            self.y_hat = F.softmax(data['y_hat'], dim=1)[:, 1] # apply softmax to get prob.
        else:
            self.y_hat = data['y_hat'] # directly get prob.
        if type(self.y) == torch.Tensor:
            self.y = self.y.detach().cpu().squeeze().numpy() # [N, ]
        elif type(self.y) == np.ndarray:
            self.y = np.squeeze(self.y) # [N, ]
        
        if type(self.y_hat) == torch.Tensor:
            self.y_hat = self.y_hat.detach().cpu().squeeze().numpy() # [N, ]
        elif type(self.y_hat) == np.ndarray:
            self.y_hat = np.squeeze(self.y_hat) # [N, ]
        
        if type(self.y_hat_full) == torch.Tensor:
            self.y_hat_full = self.y_hat_full.detach().cpu().squeeze().numpy() # [N, 2]
        elif type(self.y_hat_full) == np.ndarray:
            self.y_hat_full = np.squeeze(self.y_hat_full)

        assert self.y.shape[0] == self.y_hat.shape[0]

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y, self.y_hat, pos_label=self.pos_label, drop_intermediate=False)
        self.fpr_optimal, self.tpr_optimal, self.threshold_optimal = self._optimal_thresh(self.fpr, self.tpr, self.thresholds)

        # [n_bins, ] / [n_bins, ]
        self.cali_y, self.cali_yhat = calibration_curve(self.y, self.y_hat, n_bins=10)

    def _loss(self):
        if len(self.y_hat_full.shape) == 2: # a raw output (logit)
            with torch.no_grad():
                loss_func = torch.nn.CrossEntropyLoss()
                val_loss = loss_func(torch.FloatTensor(self.y_hat_full), torch.LongTensor(self.y))
        else: # a prob output
            with torch.no_grad():
                val_loss = F.binary_cross_entropy(torch.FloatTensor(self.y_hat_full), torch.FloatTensor(self.y))
        return val_loss.item()

    @staticmethod
    def _optimal_thresh(fpr, tpr, thresholds, p=0):
        loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
        idx = np.argmin(loss, axis=0)
        return fpr[idx], tpr[idx], thresholds[idx]

    def _auc(self):
        return metrics.auc(self.fpr, self.tpr)

    def _acc(self, threshold=None):
        if threshold is None:
            threshold = self.threshold_optimal
        pred_logit = self.y_hat > threshold
        pred_logit = pred_logit.astype(np.long)
        acc = np.sum(pred_logit == self.y) / self.y.shape[0]
        return acc

    def _recall(self, threshold=None):
        if threshold is None:
            threshold = self.threshold_optimal
        pred_logit = self.y_hat > threshold
        pred_logit = pred_logit.astype(np.long)
        recall = np.sum(pred_logit[self.y==1]) / np.sum(self.y)
        return recall

    def _precision(self, threshold=None):
        if threshold is None:
            threshold = self.threshold_optimal
        pred_logit = self.y_hat > threshold
        pred_logit = pred_logit.astype(np.long)
        precision = np.sum(self.y[pred_logit==1]) / np.sum(pred_logit)
        return precision

    def _f1_score(self, threshold=None):
        if threshold is None:
            threshold = self.threshold_optimal
        rec = self._recall(threshold)
        pre = self._precision(threshold)
        return 2 * rec * pre / (rec + pre)

    def _f1_score_mid_threshold(self):
        return self._f1_score(threshold=0.5)

    def _acc_best(self):
        best_acc = 0
        for thre in self.thresholds:
            acc = self._acc(thre)
            if acc > best_acc:
                best_acc = acc
        return best_acc

    def _acc_mid_threshold(self):
        return self._acc(threshold=0.5)

    def _ece(self):
        """Estimated Calibration Error
        """
        return np.abs(self.cali_y - self.cali_yhat).mean()

    def _mce(self):
        """Max Calibration Error
        """
        return np.abs(self.cali_y - self.cali_yhat).max()

    def compute(self, data, metrics, loss_function_name):

        # Custom loss calculation
        loss_function = get_loss_function(loss_function_name)
        loss = loss_function(data['y_hat'], data['y']).item()

        self._check_metrics(metrics)
        self._pre_compute(data)
        res_metrics = dict()
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()

        if 'loss' in metrics:
            res_metrics['loss'] = loss

        return res_metrics


class MultiClf_Evaluator(object):
    """Performance evaluator for multi-classification model"""
    def __init__(self, pred_logit=True, **kws):
        super(MultiClf_Evaluator, self).__init__()
        self.kws = kws
        self.pred_logit = pred_logit
        self.valid_functions = {
            'auc': self._auc,
            'acc': self._acc,
            'loss': self._loss,
            'macro_f1_score': partial(self._f1_score, average='macro'),
            'micro_f1_score': partial(self._f1_score, average='micro'),
        }
        self.valid_metrics = ['auc', 'loss', 'acc', 'macro_f1_score', 'micro_f1_score']
        print("[setup] A multi-classification evaluator is loaded.")
        if self.pred_logit:
            print("[warning] The input 'y_hat' will be taken as the logit output by NN.")
        else:
            print("[warning] The input 'y_hat' will be taken as multi-class probabilities.")

    def _check_metrics(self, metrics):
        for m in metrics:
            assert m in self.valid_metrics

    def ret_pred_and_gt(self):
        return self.y_hat, self.y

    def _pre_compute(self, data):
        """
        If data['y_hat'] is with a shape of [N, num_cls], recognized as a raw output (logit) from NN.
        """
        self.y, self.y_hat_full = data['y'], data['y_hat']
        assert len(self.y_hat_full.shape) > 1 and self.y_hat_full.shape[-1] > 2,\
            "Please check if it is a multi-class prediction."
        self.num_class = self.y_hat_full.shape[-1]

        if self.pred_logit:
            print("[warning] Transforming 'y_hat' into multi-class probability.")
            self.y_hat = F.softmax(data['y_hat'], dim=-1) # apply softmax to get prob.
        else:
            self.y_hat = data['y_hat'] # directly get prob.
        if type(self.y) == torch.Tensor:
            self.y = self.y.detach().cpu().squeeze().numpy() # [N, ]
        elif type(self.y) == np.ndarray:
            self.y = np.squeeze(self.y) # [N, ]
        
        # y_hat --> multi-class prob
        if type(self.y_hat) == torch.Tensor:
            self.y_hat = self.y_hat.detach().cpu().squeeze().numpy() # [N, num_class]
        elif type(self.y_hat) == np.ndarray:
            self.y_hat = np.squeeze(self.y_hat) # [N, num_class]
        # check if the summary of the last dim is 1
        assert (np.abs(np.sum(self.y_hat, axis=-1).squeeze() - 1) < 1e-5).all(), "The input y_hat cannot be summaried to 1."
        
        # y_hat_full --> from raw input (could be logit or prob)
        if type(self.y_hat_full) == torch.Tensor:
            self.y_hat_full = self.y_hat_full.detach().cpu().squeeze().numpy() # [N, num_class]
        elif type(self.y_hat_full) == np.ndarray:
            self.y_hat_full = np.squeeze(self.y_hat_full) # [N, num_class]

        assert self.y.shape[0] == self.y_hat.shape[0]

    def _loss(self):
        assert self.num_class > 2
        if self.pred_logit: # a logit output
            with torch.no_grad():
                loss_func = torch.nn.CrossEntropyLoss()
                val_loss = loss_func(torch.FloatTensor(self.y_hat_full), torch.LongTensor(self.y))
        else: # a prob output
            with torch.no_grad():
                loss_func = torch.nn.NLLLoss()
                val_loss = loss_func(torch.FloatTensor(self.y_hat_full), torch.LongTensor(self.y))
        return val_loss.item()

    def _auc(self):
        return metrics.roc_auc_score(self.y, self.y_hat, multi_class='ovr', average='macro')

    def _acc(self):
        pred_cls = np.argmax(self.y_hat, axis=-1).astype(np.long)
        acc = np.sum(pred_cls == self.y) / self.y.shape[0]
        return acc

    def _f1_score(self, average='macro'):
        assert average in ['macro', 'micro']
        pred_y = np.argmax(self.y_hat, axis=-1).astype(np.long)
        return metrics.f1_score(self.y, pred_y, average=average)

    def compute(self, data, metrics, loss_function_name):

        # Custom loss calculation
        loss_function = get_loss_function(loss_function_name)
        loss = loss_function(data['y_hat'], data['y']).item()

        self._check_metrics(metrics)
        self._pre_compute(data)
        res_metrics = dict()
        for m in metrics:
            res_metrics[m] = self.valid_functions[m]()

        if 'loss' in metrics:
            res_metrics['loss'] = loss

        return res_metrics
