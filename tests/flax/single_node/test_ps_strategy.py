import pytest
import jax.numpy as jnp

import ray
from ray.util.sgd.utils import AverageMeterCollection

from tests.flax_util import make_flax_ps_strategy


class Test_ps_strategy_single_node_2workers_2server:
    num_worker = 2
    num_ps = 2

    def setup_class(self):
        num_worker = self.num_worker
        num_ps = self.num_ps
        world_size = num_worker + num_ps
        ray.init(num_gpus=world_size, num_cpus=world_size * 3)
        self.strategy = make_flax_ps_strategy(num_ps, num_worker)

    def teardown_class(self):
        del self.strategy
        ray.shutdown()

    def test_init_strategy(self):
        self._check_sync_params()

    def _check_sync_params(self):
        strategy = self.strategy

        rets = [
            actor.get_named_parameters.remote(cpu=True)
            for actor in strategy.worker_group.actors
        ]

        params = ray.get(rets)

        keys = params[0].keys()
        num_replica = len(params)
        for key in keys:
            for i in range(num_replica - 1):
                self._assert_allclose(params[i][key], params[i + 1][key])

    @pytest.mark.parametrize("num_steps", [None, 2, 10])
    def test_train(self, num_steps):
        self.strategy.train(num_steps)
        self._check_sync_params()

    def test_validate(self):
        self.strategy.validate()

    def test_validate_result(self):
        """Make sure all workers validate results are the same.

        But it cant be test after using distributed sampler.
        """
        strategy = self.strategy

        steps = strategy.worker_group.get_data_loader_len(training=False)
        metrics = [
            AverageMeterCollection()
            for _ in range(len(strategy.worker_group.actors))
        ]

        strategy.worker_group.make_iterator(training=False)
        for idx in range(steps):
            batch_metrics = strategy.worker_group.validate_batch()
            for metric_idx, metric in enumerate(batch_metrics):
                num_sample = metric.pop("num_sample")
                metrics[metric_idx].update(metric, n=num_sample)

        keys = ["num_sample", "val_loss", "val_accuracy"]
        num_replica = len(metrics)
        for key in keys:
            for i in range(num_replica - 1):
                assert metrics[i]._meters[key].avg - \
                       metrics[i + 1]._meters[key].avg < 1e-4

    def test_states(self):
        def _assert_states(state_dict1, state_dict2):
            state_dict1 = traverse_util.flatten_dict(state_dict1["target"])
            state_dict2 = traverse_util.flatten_dict(state_dict2["target"])
            assert state_dict1.keys() == state_dict2.keys()

            for key in state_dict1.keys():
                for idx in range(len(state_dict1[key])):
                    self._assert_allclose(state_dict1[key][idx],
                                          state_dict2[key][idx])

        strategy = self.strategy
        checkpoint = "test_ps_strategy_states.pkl"

        strategy.train(1)
        states1 = strategy.get_states()

        strategy.save_states(checkpoint=checkpoint)

        strategy.train(1)  # make states different.
        strategy.load_states(checkpoint=checkpoint)
        states2 = strategy.get_states()

        _assert_states(states1["state_dict"], states2["state_dict"])

        strategy.train(1)  # make states different.
        strategy.load_states(states=states1)
        states3 = strategy.get_states()

        _assert_states(states1["state_dict"], states2["state_dict"])

    def _assert_shape(self, p, q):
        shape1 = p.shape
        shape2 = q.shape

        assert shape1 == shape2, "Input {} and {} have different shape." \
                                 "Got {} and {}.".format(p, q, shape1, shape2)

    def _assert_allclose(self, p, q):
        self._assert_shape(p, q)
        assert jnp.allclose(p, q)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
