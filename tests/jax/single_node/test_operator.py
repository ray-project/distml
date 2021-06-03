import copy

import pytest
import cupy as cp
import numpy as np

import jax
import jax.numpy as jnp

from tests.jax_util import ToyOperator
from jax.tree_util import tree_flatten
from jax._src.util import unzip2
from jax.experimental.optimizers import OptimizerState


class Test_jax_operator:
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

        assert operator.opt_state
        assert operator.init_fun
        assert operator.predict_fun

        assert operator.opt_init
        assert operator.opt_update
        assert operator.get_params

        assert operator.criterion

    def test_update(self):
        operator = self.operator

        iterator = iter(operator._train_loader)

        batch = next(iterator)

        loss_val, grads = operator.derive_updates(batch)

        assert isinstance(loss_val, float)
        assert isinstance(grads, dict)

        assert (len(grads) ==
                len(operator.get_parameters(cpu=True)))

        operator.apply_updates(grads)

    def test_states(self):
        operator = self.operator
        states = operator.get_states()
        states = copy.deepcopy(states)
        params = operator.get_parameters(cpu=True)
        params = copy.deepcopy(params)

        states_flat, tree, subtrees = operator.opt_state
        new_states_flat, new_subtrees = unzip2(map(tree_flatten, states["opt_state"]))
        new_opt_state = OptimizerState(new_states_flat, tree, new_subtrees)
        params_1 = operator.get_params(new_opt_state)
        params_1, _ = tree_flatten(params_1)

        for idx in range(len(params)):
            self._assert_allclose(
                params[idx], params_1[idx])

        tmp_state_path = "tmp_states.pkl"
        operator.save_states(tmp_state_path)
        iterator = iter(operator._train_loader)

        def train_batch():
            batch = next(iterator)
            loss_val, grads = operator.derive_updates(batch)
            operator.apply_updates(grads)

        train_batch()
        batch = next(iterator)
        loss_val, grads = operator.derive_updates(batch)
        operator.apply_updates(grads)

        with pytest.raises(AssertionError):
            params_2 = operator.get_parameters(cpu=True)
            self._assert_allclose(
                params[0], params_2[0])

        operator.load_states(checkpoint=tmp_state_path)
        params_2 = operator.get_parameters(cpu=True)

        for idx in range(len(params)):
            self._assert_allclose(
                params[idx], params_2[idx])

        train_batch()
        operator.load_states(states=states)
        params_3 = operator.get_parameters(cpu=True)

        for idx in range(len(params)):
            self._assert_allclose(
                params[idx], params_3[idx])

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
                self._assert_allclose(
                    custom_states[key], custom_states_2[key]
                )

    @pytest.mark.parametrize("array_shape",
                             [(1,), (3, 3), (1, 1, 1), (3, 3, 3), (3, 3, 3, 3)])
    def test_to_cupy(self, array_shape):
        operator = self.operator

        tensor = np.random.rand(*array_shape)
        tensor = jax.device_put(tensor)
        cupy_tensor = operator.to_cupy(tensor)

        assert isinstance(cupy_tensor, cp.ndarray)
        assert (cupy_tensor.data.ptr ==
                tensor.device_buffer.unsafe_buffer_pointer())

    @pytest.mark.parametrize("array_shape",
                             [(1,), (3, 3), (1, 1, 1), (3, 3, 3), (3, 3, 3, 3)])
    def test_get_jax_dlpack(self, array_shape):
        operator = self.operator

        tensor = np.random.rand(*array_shape)
        tensor = cp.asarray(tensor)
        jax_tensor = operator.to_operator_tensor(tensor)

        assert isinstance(jax_tensor, jnp.ndarray)
        assert (tensor.data.ptr ==
                jax_tensor.device_buffer.unsafe_buffer_pointer())

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

        operator.reset_optimizer_for_params(new_params)

        params_2 = operator.get_named_parameters(cpu=True)

        assert params_2.keys() == new_params.keys()

        for key in params_2.keys():
            self._assert_allclose(
                params_2[key], new_params[key])

    def test_clean_redundancy(self):
        operator = self.operator

        assert operator._train_loader
        assert operator._validation_loader

        operator.clean_redundancy()

        assert not operator._train_loader
        assert not operator._validation_loader

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

    sys.exit(pytest.main(["-v", "-x", "-s", __file__]))
