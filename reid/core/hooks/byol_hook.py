from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class BYOLHook(Hook):
    """Hook for BYOL.
    """

    def __init__(self, end_momentum=1.):
        self.end_momentum = end_momentum

    def _get_model(self, runner):
        if is_module_wrapper(runner.model):
            return runner.model.module
        else:
            return runner.model

    def before_train_iter(self, runner):
        model = self._get_model(runner)
        cur_iter = runner.iter
        max_iters = runner.max_iters
        base_m = model.base_momentum
        m = self.end_momentum - (self.end_momentum - base_m) * (
            cos(pi * cur_iter / float(max_iters)) + 1) / 2
        model.momentum = m

    def after_train_iter(self, runner):
        model = self._get_model(runner)
        model.momentum_update()
