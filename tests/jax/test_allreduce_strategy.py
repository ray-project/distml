"""Test the send/recv API."""
import pytest
import numpy as np
import os

import ray
from ray.util.collective.types import Backend
from ray.util.collective.tests.conftest import clean_up

from distml.tests.torch_util import make_jax_ar_strategy

import jax
import jax.numpy as jnp


class Test_allreduce_strategy_single_node_2workers:
    world_size = 2
    def setup_class(self):
        world_size = self.world_size
        ray.init(num_gpus=world_size,
                 num_cpus=world_size*2)
        self.strategy = make_jax_ar_strategy(world_size)

    def teardown_class(self):
        del self.strategy
        os.system("ray stop")


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
