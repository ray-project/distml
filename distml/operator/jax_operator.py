from typing import Any, Mapping, Optional

import numpy as np
import cupy as cp

import jax
from jax import value_and_grad
import jax.numpy as jnp
from jax.lib import xla_client
from jax.dlpack import from_dlpack
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure
from jax._src.util import unzip2
from jax.experimental.optimizers import OptimizerState

from distml.operator.base_operator import TrainingOperator


class JAXTrainingOperator(TrainingOperator):
    def __init__(self, operator_config: Optional[Mapping[str, Any]]):
        super(JAXTrainingOperator, self).__init__(operator_config)
        # Should be set by users in the `register` function.
        # model methods
        self.opt_state = None
        self.init_fun = None
        self.predict_fun = None
        # optimizer methods
        self.opt_init = None
        self.opt_update = None
        self.get_params = None

        self.criterion = None

        # Data loaders for training and validation, registered by users.
        self._train_loader = None
        self._validation_loader = None

        self.setup(operator_config)

        if hasattr(operator_config, "jit_mode"):
            if operator_config["jit_mode"]:
                raise NotImplementedError("Not support jit in jax operator.")

        self.train_step_num = 0

    def setup(self, *args, **kwargs):
        """Function that needs to be override by users.

        Example:
            # some code is the same for all users,
            # maybe we can put it in register.
            rng_key = random.PRNGKey(0)
            input_shape = (28, 28, 1, 64)
            lr=0.01
            init_fun, predict_fun = ResNet18(num_classes)
            _, init_params = init_fun(rng_key, input_shape)

            opt_init, opt_update, get_params = optimizers.adam(lr)
            opt_state = opt_init(init_params)

            criterion = lambda logits, targets:-jnp.sum(logits * targets)

            self.register(model=(opt_state, init_fun, predict_fun),
                          optimizer=(opt_init, opt_update, get_params),
                          criterion=criterion)
        """
        raise NotImplementedError("Please override this function to register "
                                  "your model, optimizer, and criterion.")

    def register(self, *, model, optimizer, criterion, jit_mode: bool = False):
        """Register a few critical information about the model to operator.

        Args:
            model (tuple/list): a tuple/list has three elements. The first
                element should be opt_states that return from opt_init.The
                second element should be init_fun that used to initialize
                model params. The third element should be predict_fun that
                feed params and inputs, return prediction.
            optimizer (tuple/list): a tuple/list has three elements. The
                first element should be opt_init that used to initialize
                optimizer state. The second element should be opt_update
                that use to update the optimizer state. The third element
                should be get_params that feed opt_states and return the
                params.
            criterion (function): a function use to calculate the loss value.
            jit_mode (bool): use the jit mode in jax.
        """

        if not isinstance(model, (tuple, list)) and len(model) != 3:
            raise RuntimeError("`model` must be a tuple or list and contains"
                               "'opt_states', 'init_fun', 'predict_fun'."
                               "Got: {} {}".format(type(model), len(model)))

        if not isinstance(optimizer, (tuple, list)) and len(optimizer) != 3:
            raise RuntimeError(
                "`optimizer` must be a tuple or list and contains"
                "'opt_init', 'opt_update' and 'get_params'."
                "Got: {} {}".format(type(optimizer), len(optimizer)))

        if not hasattr(criterion, "__call__"):
            raise RuntimeError(
                "The `criterion` must be callable function that "
                "feed logits and target, return the loss value. "
                "Got: {}".format(type(criterion)))

        self.criterion = criterion
        self._register_model(model)
        self._register_optimizer(optimizer)

    def _register_model(self, model):
        """register model components."""

        if not isinstance(model[0],
                          jax.experimental.optimizers.OptimizerState):
            raise RuntimeError(
                "The first elemente of `model` must be the "
                "`opt_states` return from optimizer `opt_init`. "
                "Got: {}".format(type(model[0])))

        if not hasattr(model[1], "__call__"):
            raise RuntimeError("The second elemente of `model` must be the "
                               "`init_fun` return from model. "
                               "Got: {}".format(type(model[1])))

        if not hasattr(model[2], "__call__"):
            raise RuntimeError("The third elemente of `model` must be the "
                               "`predict_fun` return from model. "
                               "Got: {}".format(type(model[2])))

        self.opt_state = model[0]
        self.init_fun = model[1]
        self.predict_fun = model[2]

    def _register_optimizer(self, optimizer):
        """register optimizer components."""
        if not hasattr(optimizer[0], "__call__"):
            raise RuntimeError("The fist elemente of `optimizer` must be the "
                               "`opt_init` return from optimizer. "
                               "Got: {}".format(type(optimizer[1])))

        if not hasattr(optimizer[1], "__call__"):
            raise RuntimeError(
                "The second elemente of `optimizer` must be the "
                "`opt_update` return from optimizer. "
                "Got: {}".format(type(optimizer[1])))

        if not hasattr(optimizer[2], "__call__"):
            raise RuntimeError("The third elemente of `optimizer` must be the "
                               "`get_params` return from optimizer. "
                               "Got: {}".format(type(optimizer[2])))

        self.opt_init = optimizer[0]
        self.opt_update = optimizer[1]
        self.get_params = optimizer[2]

    def register_data(self, *, train_loader=None, validation_loader=None):
        self._train_loader = train_loader
        self._validation_loader = validation_loader

    def _get_train_loader(self):
        return self._train_loader

    def _get_validation_loader(self):
        return self._validation_loader

    def loss_func(self, params, batch):
        """A function to calculate predictions and loss value.

        This function is going to be decorated by
        `grad` in Jax to calculate gradients.

        Args:
            params (list): The params return from get_params(opt_states).
            batch (tuple): a data batch containing a feature/target pair.
        """
        inputs, targets = batch
        logits = self.predict_fun(params, inputs)
        return self.criterion(logits, targets)

    def derive_updates(self, batch):
        """Compute the parameter updates on a given batch of data.

        The `derive_updates` function should be called in conjunction with
        the next `apply_updates` function in order to finish one iteration
        of training.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        loss_val, gradient = self._calculate_gradient(self.opt_state, batch)

        gradient_dict, tree = tree_flatten(gradient)
        assert tree == self.opt_state[1]

        if hasattr(self, "preset_keys"):
            gradient_dict = {
                k: g
                for k, g in zip(self.preset_keys, gradient_dict)
            }
        else:
            gradient_dict = {
                f"{idx}": g
                for idx, g in enumerate(gradient_dict)
            }
        return loss_val.item(), gradient_dict

    def apply_updates(self, updates):
        """Set and apply the updates using the opt_update in Jax.

        Args:
            updates (dict): a dictionary of parameter name and updates.
        """
        keys, updates = unzip2(
            sorted(updates.items(), key=lambda d: int(d[0])))
        updates = tree_unflatten(self.opt_state[1], updates)
        self.opt_state = self.opt_update(self.train_step_num, updates,
                                         self.opt_state)
        self.train_step_num += 1

    def to_cupy(self, tensor):
        """Convert a jax GPU tensor to cupy tensor."""
        if isinstance(tensor, list):
            return list(map(self.to_cupy, tensor))
        ctensor = cp.fromDlpack(self.get_jax_dlpack(tensor))
        return ctensor

    def to_operator_tensor(self, tensor):
        """Convert a cupy tensor to jax tensor.

        There comes a bug. The layouts of tensor explained by cupy
        and jax are different. But dlpack doesn't convert the layout.
        """
        if isinstance(tensor, list):
            return list(map(self.to_operator_tensor, tensor))
        return from_dlpack(tensor.toDlpack())

    # TODO(HUI): support return logits by adding use_aux in `value_and_grad`
    def _calculate_gradient(self, opt_state, batch):
        params = self.get_params(opt_state)
        loss_val, gradient = value_and_grad(self.loss_func)(params, batch)
        return loss_val, gradient

    def get_jax_dlpack(self, tensor):
        """Get the dlpack of a jax tensor.

        Jax api might cause different pointer address after the conversion.
        We use the xla api to avoid this bug.
        """
        return xla_client._xla.buffer_to_dlpack_managed_tensor(
            tensor.device_buffer, take_ownership=False)

    def validate_batch(self, batch):
        """Perform validation over a data batch.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        params = self.get_params(self.opt_state)
        criterion = self.criterion
        predict_fun = self.predict_fun

        # unpack features into list to support multiple inputs model
        features, targets = batch

        outputs = predict_fun(params, features)
        loss = criterion(outputs, targets)
        prediction_class = jnp.argmax(outputs, axis=1)
        targets_class = jnp.argmax(targets, axis=1)

        acc = jnp.mean(prediction_class == targets_class)
        samples_num = targets.shape[0]

        return {
            "val_loss": loss.item(),
            "val_accuracy": acc.item(),
            "samples_num": samples_num
        }

    def get_parameters(self, cpu: bool):
        """get the flatten parameters."""
        params = self.get_params(self.opt_state)
        flatten_params, tree = tree_flatten(params)
        if not hasattr(self, "tree"):
            self.tree = tree

        if cpu:
            flatten_params = list(map(np.asarray, flatten_params))
        return flatten_params

    def get_named_parameters(self, cpu: bool):
        """Get the named parameters.

        In jax, we need to construct a dict to contain the parameters.
        """
        params = self.get_parameters(cpu)
        if hasattr(self, "preset_keys"):
            dict_params = {
                name: p
                for name, p in zip(self.preset_keys, params)
            }
        else:
            dict_params = {f"{idx}": p for idx, p in enumerate(params)}

        return dict_params

    # TODO(HUI): used in load states or load parameters
    def set_parameters(self, new_params):
        """Use new parameters to replace model parameters.

        In jax, we need to construct a dict to contain the parameters.

        Args:
            new_params (dict): New parameters to updates the current model.
        """
        assert isinstance(new_params, dict)

        # make sure all params in GPU. Should be controlled of use_gpu.
        new_params = {k: jax.device_put(v) for k, v in new_params.items()}

        keys, new_params = unzip2(
            sorted(new_params.items(), key=lambda d: int(d[0])))
        self.preset_keys = keys

        if not hasattr(self, "tree"):
            self.tree = tree_structure(self.get_params(self.opt_state))

        states_flat, tree, subtrees = self.opt_state

        states = map(tree_unflatten, subtrees, states_flat)

        def update(param, state):
            new_state = param, *state[1:]
            return new_state

        new_states = map(update, new_params, states)

        new_states_flat, new_subtrees = unzip2(map(tree_flatten, new_states))

        if not new_subtrees:
            raise RuntimeError("subtrees of new params is empty.")
        for idx, (subtree, new_subtree) in enumerate(
                zip(subtrees, new_subtrees)):
            if new_subtree != subtree:
                msg = (
                    "input structure did not match the save params structure. "
                    "input {} and output {}.")
                raise TypeError(msg.format(subtree, new_subtree))

        self.opt_state = OptimizerState(new_states_flat, tree, new_subtrees)

    def reset_optimizer_for_params(self, params):
        if not isinstance(params, dict):
            raise RuntimeError("The `params` should be dict. "
                               "Got {}".format(type(params)))

        keys, params = unzip2(sorted(params.items(), key=lambda d: int(d[0])))
        self.tree = tree_structure(params)
        self.opt_state = self.opt_init(params)

    def ones(self, shape, cpu: bool = True):
        if cpu:
            return np.ones(shape)
        else:
            return jnp.ones(shape)

    def zeros(self, shape, cpu: bool = True):
        if cpu:
            return np.zeros(shape)
        else:
            return jnp.zeros(shape)

    def ones_like(self, x, cpu: bool = True):
        if cpu:
            return np.ones_like(x)
        else:
            return jnp.ones_like(x)

    def zeros_like(self, x, cpu: bool = True):
        if cpu:
            return np.zeros_like(x)
        else:
            return jnp.zeros_like(x)

    def numel(self, v):
        return np.size(v)

    def asarray(self, v):
        return jnp.asarray(v)

    def clean_redundancy(self):
        if self._train_loader:
            del self._train_loader
            self._train_loader = None
        if self._validation_loader:
            del self._validation_loader
            self._validation_loader = None

    # TODO(HUI): use pickle to serialize parameters or states and save it.
    def save_parameters(self, checkpoint: str):
        raise NotImplementedError(
            "save_parameters is not support in jax operator.")

    def load_parameters(self, checkpoint: str):
        raise NotImplementedError(
            "load_parameters is not support in jax operator.")

    def save_states(self, checkpoint: str):
        raise NotImplementedError(
            "save_states is not support in jax operator.")

    def get_states(self):
        raise NotImplementedError("get_states is not support in jax operator.")

    def load_states(self, checkpoint: str):
        raise NotImplementedError(
            "load_states is not support in jax operator.")
