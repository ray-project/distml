import pytest
import cupy as cp
import numpy as np
import torch

from tests.torch_util import ToyOperator


class Test_torch_operator:
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
        assert operator._model
        assert operator._optimizer
        assert operator._criterion

    def test_update(self):
        operator = self.operator

        iterator = iter(operator._train_loader)

        batch = next(iterator)

        loss_val, grads = operator.derive_updates(batch)

        assert isinstance(loss_val, float)
        assert isinstance(grads, dict)

        assert (len(grads) == len(operator.get_states()["model"]))

        operator.apply_updates(grads)

    def test_states(self):
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
                states["model"][key].cpu(),
                states_2["model"][key].cpu(),
                atol=0.01,
                rtol=0.1)

        train_batch()
        operator.load_states(states=states)
        states_3 = operator.get_states()

        for key in states["model"].keys():
            assert np.allclose(
                states["model"][key].cpu(),
                states_3["model"][key].cpu(),
                atol=0.01,
                rtol=0.1)

    def test_to_cupy(self):
        operator = self.operator

        tensor = torch.ones(10, 10)
        cupy_tensor = operator.to_cupy(tensor)

        assert isinstance(cupy_tensor, torch.Tensor)
        with pytest.raises(TypeError):
            if not isinstance(cupy_tensor, cp.ndarray):
                raise TypeError("Tensor is not cupy array")


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
