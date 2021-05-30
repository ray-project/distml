"""Test the send/recv API."""
import pytest
import numpy as np
import os

import ray
from ray.util.collective.types import Backend
from ray.util.collective.tests.conftest import clean_up
from ray.util.sgd.utils import AverageMeterCollection

from tests.torch_util import make_torch_ar_strategy

import torch


class Test_allreduce_strategy_single_node_2workers:
    world_size = 2

    def setup_class(self):
        world_size = self.world_size
        ray.init(num_gpus=world_size,
                 num_cpus=world_size * 2)
        self.strategy = make_torch_ar_strategy(world_size)

    def teardown_class(self):
        del self.strategy

    def test_init_strategy(self):
        self._check_sync_params()

    def _check_sync_params(self):
        strategy = self.strategy

        rets = [replica.get_named_parameters.remote(cpu=True)
                for replica in strategy.data_parallel_group.replicas]

        params = ray.get(rets)

        keys = params[0].keys()
        num_replica = len(params)
        for key in keys:
            for i in range(num_replica - 1):
                self._assert_allclose(params[i][key],
                                      params[i+1][key])

    @pytest.mark.parametrize("num_steps", [None, 2, 10])
    def test_train(self, num_steps):
        self.strategy.train(num_steps) # train a full epoch
        self._check_sync_params()

    def test_validate(self):
        metrics = self.strategy.validate()

    def test_validate_result(self):
        """Make sure all replicas validate results are the same.

        But it cant be test after using distributed sampler.
        """
        strategy = self.strategy

        steps = strategy.data_parallel_group.get_data_loader_len(training=False)
        metrics = [AverageMeterCollection()
                   for _ in range(len(strategy.data_parallel_group.replicas))]

        strategy.data_parallel_group.make_iterator(training=False)
        for idx in range(steps):
            batch_metrics = strategy.data_parallel_group.validate_batch()
            for metric_idx, metric in enumerate(batch_metrics):
                num_sample = metric.pop("num_sample")
                metrics[metric_idx].update(metric, n=num_sample)

        keys = ["num_sample", "val_loss", "val_accuracy"]
        num_replica = len(metrics)
        for key in keys:
            for i in range(num_replica - 1):
                assert metrics[i]._meters[key].avg - \
                       metrics[i+1]._meters[key].avg < 1e-4

    def _assert_shape(self, p, q):
        shape1 = p.shape
        shape2 = q.shape

        assert shape1 == shape2, "Input {} and {} have different shape." \
                                 "Got {} and {}.".format(p, q, shape1, shape2)

    def _assert_allclose(self, p, q):
        if isinstance(p, torch.Tensor):
            p = p.cpu().numpy()
        if isinstance(q, torch.Tensor):
            q = q.cpu().numpy()

        self._assert_shape(p, q)
        assert np.allclose(p, q)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__])) #  ,"-s" # for debug
