from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MMCLHook(Hook):

    def before_train_epoch(self, runner):
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        model.set_epoch(runner.epoch)
        m = model.base_momentum * runner.epoch / runner.max_epochs
        model.momentum = m
