"""Test the send/recv API."""
import pytest
import cupy as cp
import numpy as np

import jax
import jax.numpy as jnp

from tests.jax_util import ToyOperator


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

        with pytest.raises(NotImplementedError):
            assert len(grads) == \
                   len(operator.get_states()["model"])

        operator.apply_updates(grads)

    def tmp_test_states(self):
        operator = self.operator
        states = operator.get_states()

        tmp_state_path = "tmp_states.pth"
        operator.save_states(tmp_state_path)

        def train_batch():
            iterator = iter(operator._train_loader)
            batch = next(iterator)
            loss_val, grads = operator.derive_updates(batch)
            operator.apply_updates(grads)

        train_batch()
        operator.load_states(checkpoint=tmp_state_path)
        states_2 = operator.get_states()

        for key in states["model"].keys():
            assert np.allclose(
                states["model"][key].cpu(), states_2["model"][key].cpu(),
                atol=0.01, rtol=0.1)

        train_batch()
        operator.load_states(states=states)
        states_3 = operator.get_states()

        for key in states["model"].keys():
            assert np.allclose(
                states["model"][key].cpu(), states_3["model"][key].cpu(),
                atol=0.01, rtol=0.1)

    @pytest.mark.parametrize("array_shape",
                             [(1,), (3, 3), (1, 1, 1), (3, 3, 3), (3, 3, 3, 3)])
    def test_to_cupy(self, array_shape):
        operator = self.operator

        tensor = np.random.rand(*array_shape)
        tensor = jax.device_put(tensor)
        cupy_tensor = operator.to_cupy(tensor)

        assert isinstance(cupy_tensor, cp.ndarray)
        assert cupy_tensor.data.ptr == tensor.device_buffer.unsafe_buffer_pointer()

    @pytest.mark.parametrize("array_shape",
                             [(1,), (3, 3), (1, 1, 1), (3, 3, 3), (3, 3, 3, 3)])
    def test_get_jax_dlpack(self, array_shape):
        operator = self.operator

        tensor = np.random.rand(*array_shape)
        tensor = cp.asarray(tensor)
        jax_tensor = operator.to_operator_tensor(tensor)

        assert isinstance(jax_tensor, jnp.ndarray)
        assert tensor.data.ptr == jax_tensor.device_buffer.unsafe_buffer_pointer()

    def test_clean_redundancy(self):
        operator = self.operator

        assert operator._train_loader
        assert operator._validation_loader

        operator.clean_redundancy()

        assert not operator._train_loader
        assert not operator._validation_loader

if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
