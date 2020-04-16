# Copyright (c) Open-MMLab. All rights reserved.
from ..dist_utils import master_only
from .hook import HOOKS, Hook


@HOOKS.register_module
class CheckpointHook(Hook):

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

    @master_only
    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        runner.logger.info(f'Save checkpoint at {runner.iter} iterations')
