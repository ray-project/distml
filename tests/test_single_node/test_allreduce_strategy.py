"""Test the send/recv API."""
import pytest
import numpy as np
import ray

from ray.util.collective.types import Backend

from distml.tests.torch_util import make_torch_ar_strategy


# multi-group strategy training
# allreduce basic training
@pytest.mark.parametrize("backend", [Backend.NCCL])
def test_reduce_different_name(ray_start_single_node, backend, group_name="default"):
    strategy = make_torch_ar_strategy()
    strategy.data_parallel_group.make_iterator()

    def _check_iterable(target):
        assert hasattr(target, '__iter__')

    strategy.data_parallel_group.apply(_check_iterable)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
