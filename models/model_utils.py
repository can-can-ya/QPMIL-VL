import numpy as np


def freeze_weight(model, model_name):
    for param in model.parameters():
        param.requires_grad = False
    print('[setup] weights of {} are frozen.'.format(model_name))


def set_tunable_v(tunable_v, task_num):
    for param in tunable_v:
        param.requires_grad = False
        param.grad = None

    tunable_v[task_num - 1].requires_grad = True


class EarlyStopping:
    """Early stops the training if monitoring metric doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, verbose=False, threshold=1e-6):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.counter = 0
        self.early_stop = False
        self.save_checkpoint = False
        self.metric_min = np.Inf

    def __call__(self, epoch, metric):

        self.save_checkpoint = False

        if epoch <= self.warmup:
            pass
        elif self.metric_min == np.Inf:
            self.update_metric(metric)
        elif self.metric_min - metric < self.threshold:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.update_metric(metric)

    def stop(self, **kws):
        return self.early_stop

    def save_ckpt(self, **kws):
        return self.save_checkpoint

    def update_metric(self, metric):
        '''Saves model when monitoring metric decrease.'''
        if self.verbose:
            print(f'Monitoring metric decreased ({self.metric_min:.6f} --> {metric:.6f}).  Saving model ...')
        self.metric_min = metric
        self.save_checkpoint = True