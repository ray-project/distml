from flax import serialization
from numpy.lib.arraysetops import isin
import pytest
import cupy as cp
import numpy as np

import jax
import jax.numpy as jnp

from tests.flax_util import ToyOperator
from jax.tree_util import tree_flatten
from jax._src.util import unzip2
import flax.traverse_util as traverse_util

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map

import flax.traverse_util as traverse_util
from flax.core import unfreeze

class Test_flax_operator:
    def setup_class(self):
        operator_config = {
            "lr": 0.01,
            "test_mode": True,  # subset the data
            # this will be split across workers.
            "batch_size": 16
        }
        self.operator = ToyOperator(operator_config=operator_config)

    def teardown_class(self):
        del self.operator

    def test_setup(self):
        operator = self.operator
        assert operator._train_loader
        assert operator._validation_loader

        print(operator.model)
        assert operator.model
        assert operator.optimizer

        assert operator.criterion

    def test_update(self):
        operator = self.operator

        iterator = iter(operator._train_loader)

        batch = next(iterator)

        loss_val, grads = operator.derive_updates(batch)

        assert isinstance(loss_val, float)
        assert isinstance(grads, dict)

        assert (len(grads) == len(operator.get_parameters(cpu=True)))

        operator.apply_updates(grads)

    def test_states(self):
        operator = self.operator
        states = operator.get_states()
        params = operator.get_parameters(cpu=True)

        tmp_state_path = "test_operator_states.pkl"
        operator.save_states(tmp_state_path)
        iterator = iter(operator._train_loader)

        def train_batch():
            batch = next(iterator)
            loss_val, grads = operator.derive_updates(batch)
            operator.apply_updates(grads)

        train_batch()

        with pytest.raises(AssertionError):
            params_1 = operator.get_parameters(cpu=True)
            self._assert_allclose(params[0], params_1[0])

        operator.load_states(checkpoint=tmp_state_path)
        params_1 = operator.get_parameters(cpu=True)

        for idx in range(len(params)):
            self._assert_allclose(params[idx], params_1[idx])

        train_batch()
        operator.load_states(states=states)
        params_2 = operator.get_parameters(cpu=True)

        for idx in range(len(params)):
            self._assert_allclose(params[idx], params_2[idx])

    def test_load_states_with_keys(self):
        num_key = 2
        operator = self.operator

        states1 = operator.get_states()
        params = operator.get_named_parameters(cpu=True)

        new_params = {}
        new_keys = []
        num = 0
        for k, v in params.items():
            if num == num_key:
                break
            new_params[k] = jnp.zeros_like(v)
            new_keys.append(k)
            num += 1

        new_keys = tuple(new_keys)
        
        operator.reset_optimizer_for_params(new_params)
        params = operator.get_named_parameters(cpu=True)

        operator.load_states(states=states1, keys=new_keys)

        states2 = operator.get_states()

        state_dict1 = states1["state_dict"]["target"]
        state_dict2 = states2["state_dict"]["target"]

        state_dict1 = traverse_util.flatten_dict(state_dict1)
        state_dict2 = traverse_util.flatten_dict(state_dict2)

        for key in new_keys:
            for idx in range(len(state_dict1[key])):
                self._assert_allclose(state_dict1[key][idx],
                                      state_dict2[key][idx])
        
    def test_custom_states(self):
        operator = self.operator

        custom_states = {
            "random_state1": jax.device_put(np.random.rand(1)),
            "random_state2": jax.device_put(np.random.rand(10, 10)),
            "random_state3": jax.device_put(np.random.rand(10, 100)),
        }
        operator.register_custom_states(custom_states)

        state = operator.get_states()

        custom_states_2 = state.get("custom_states", None)

        if custom_states_2:
            keys = custom_states.keys()
            keys_2 = custom_states_2.keys()
            assert keys == keys_2
            for key in keys:
                self._assert_allclose(custom_states[key], custom_states_2[key])

    @pytest.mark.parametrize("num_key", [1, 3, 5])
    def test_reset_optimizer(self, num_key):
        operator = self.operator

        params = operator.get_named_parameters(cpu=True)

        new_params = {}

        num = 0
        for k, v in params.items():
            if num == num_key:
                break
            new_params[k] = v
            num += 1

        operator.reset_optimizer_for_params(new_params)

        params_2 = operator.get_named_parameters(cpu=True)

        assert params_2.keys() == new_params.keys()

        for key in params_2.keys():
            self._assert_allclose(params_2[key], new_params[key])

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
