import warnings

from mmcv.runner import Hook

from .evaluator import Evaluator


class EvalHook(Hook):

    def __init__(self, data_loader, start=None, interval=1, **eval_kwargs):
        if not interval > 0:
            raise ValueError(f'interval must be positive, but got: {interval}')
        if start is not None and start < 0:
            warnings.warn(f'evaluation start epoch {start} is smaller than 0'
                          'use 0 instead')
            start = 0

        self.interval = interval
        self.start = start

        self.initial_epoch_flag = True

        # build evaluator
        self.data_loader = data_loader
        self.evaluator = Evaluator(**eval_kwargs)

    def before_train_epoch(self, runner):
        """Evaluation the model only at the start of training."""
        if not self.initial_epoch_flag:
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_epoch_flag = False

    def evaluation_flag(self, runner):
        """Judge whether to perform evaluation after this epoch.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                return False  # No evaluation
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False
        return True

    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return
        self.evaluator.eval(
            runner.model, self.data_loader, logger=runner.logger)
