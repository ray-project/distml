import pytest
import jax.numpy as jnp

import ray
from ray.util.sgd.utils import AverageMeterCollection

from tests.jax_util import make_jax_ar_strategy


class Test_allreduce_strategy_two_node_2workers:
    world_size = 2

    def setup_class(self):
        world_size = self.world_size
        ray.init("auto")
        self.strategy = make_jax_ar_strategy(world_size)

    def teardown_class(self):
        del self.strategy
        ray.shutdown()

    def test_init_strategy(self):
        self._check_sync_params()

    def _check_sync_params(self):
        strategy = self.strategy

        rets = [
            replica.get_named_parameters.remote(cpu=True)
            for replica in strategy.data_parallel_group.replicas
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
        """Make sure all replicas validate results are the same.

        But it cant be test after using distributed sampler.
        """
        strategy = self.strategy

        steps = strategy.data_parallel_group.get_data_loader_len(
            training=False)
        metrics = [
            AverageMeterCollection()
            for _ in range(len(strategy.data_parallel_group.replicas))
        ]

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
                       metrics[i + 1]._meters[key].avg < 1e-4

    def test_states(self):
        def _assert_states(opt_state1, opt_state2):
            assert opt_state1.keys() == opt_state2.keys()

            for key in opt_state1.keys():
                for idx in range(len(opt_state1[key])):
                    self._assert_allclose(opt_state1[key][idx],
                                          opt_state2[key][idx])

        strategy = self.strategy
        checkpoint = "tmp_states.pkl"

        states1 = strategy.get_states()
        strategy.save_states(checkpoint=checkpoint)

        strategy.train(1)  # make states different.
        strategy.load_states(checkpoint=checkpoint)
        states2 = strategy.get_states()

        _assert_states(states1["opt_state"], states2["opt_state"])

        strategy.train(1)  # make states different.
        strategy.load_states(states=states1)
        states2 = strategy.get_states()

        _assert_states(states1["opt_state"], states2["opt_state"])

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
