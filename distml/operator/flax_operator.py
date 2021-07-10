import os
import pickle
import warnings

from typing import Optional

import numpy as np
from jax import grad, value_and_grad
import jax.numpy as jnp
from jax.tree_util import tree_map

import flax.traverse_util as traverse_util
from flax.core import unfreeze, freeze
from flax import serialization

from distml.operator.jax_operator import JAXTrainingOperator


class FLAXTrainingOperator(JAXTrainingOperator):

    def __init__(self, *, operator_config):
        self.model = None
        self.optimizer = None

        super(FLAXTrainingOperator, self).__init__(operator_config=operator_config)
        delattr(self, "opt_state")
        delattr(self, "init_fun")
        delattr(self, "predict_fun")
        delattr(self, "opt_init")
        delattr(self, "opt_update")
        delattr(self, "get_params")


    def register(self, 
                 *,
                 model, 
                 optimizer, 
                 criterion, 
                 lr_scheduler=None,
                 jit_mode: bool = False):
        """Register a few critical information about the model to operator."""

        self.criterion = criterion
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler
        
        self._register_model(model)
        self._register_optimizer(optimizer)
            
    def _register_model(self, model):
        self.model = model

    def _register_optimizer(self, optimizer):
        self.optimizer = optimizer

    def loss_func(self, params, batch):
        """A function to calculate predictions and loss value.

        This function is going to be decorated by
        `grad` in Jax to calculate gradients.

        Args:
            params (list): The params return from get_params(opt_states).
            batch (tuple): a data batch containing a feature/target pair.
        """
        inputs, targets = batch

        logits = self.model.apply(params, inputs)
        return self.criterion(logits, targets)

    def derive_updates(self, batch):
        """Compute the parameter updates on a given batch of data.

        The `derive_updates` function should be called in conjunction with
        the next `apply_updates` function in order to finish one iteration
        of training.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        loss_val, gradient = self._calculate_gradient(self.optimizer, batch)
        gradient_dict = traverse_util.flatten_dict(unfreeze(gradient))
        return loss_val.item(), gradient_dict

    def apply_updates(self, gradient):
        """Set and apply the updates using the apply_gradient in Flax.

        Args:
            updates (dict): a dictionary of parameter name and updates.
        """
        assert isinstance(gradient, dict)

        gradient = freeze(traverse_util.unflatten_dict(gradient))

        hyper_params = {}
        if self.lr_scheduler:
            hyper_params["learning_rate"] = self.lr_scheduler.step()

        self.optimizer = self.optimizer.apply_gradient(gradient, **hyper_params)
        self.train_step_num += 1

    def _calculate_gradient(self, optimizer, batch):
        params = optimizer.target
        
        loss_val, gradient = value_and_grad(self.loss_func)(params, batch)
        return loss_val, gradient

    def validate_batch(self, batch):
        """Perform validation over a data batch.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        params = self.optimizer.target
        criterion = self.criterion
        predict_fun = self.model.apply

        # unpack features into list to support multiple inputs model
        inputs, targets = batch

        outputs = predict_fun(params, inputs)
        loss = criterion(outputs, targets)
        prediction_class = jnp.argmax(outputs, axis=1)
        targets_class = jnp.argmax(targets, axis=1)

        acc = jnp.mean(prediction_class == targets_class)
        samples_num = targets.shape[0]
        return {
            "val_loss": loss.item(),
            "val_accuracy": acc.item(),
            "num_sample": samples_num
        }

    def _param_list2dict(self, param):
        if not hasattr(self, "parmas_keys"):
            self.parmas_keys = list(traverse_util.flatten_dict(self.optimizer.target).keys())
        param = {k:v for k,v in zip(self.parmas_keys, param)}
        return param

    def reset_optimizer_for_params(self, params):
        assert isinstance(params, dict)
        params = traverse_util.unflatten_dict(params)
        self.optimizer = self.optimizer.optimizer_def.create(freeze(params))

    def get_states(self):
        state_dict = serialization.to_state_dict(self.optimizer)

        states = {
            "state_dict": state_dict,
        }

        if self._custom_states:
            states.update({"custom": self.get_custom_states()})

        if self.lr_scheduler and hasattr(self.lr_scheduler,
                                         "get_state_dict()"):
            states.update({"lr_scheduler": self.lr_scheduler.get_state_dict()})

        return states

    def load_states(self,
                    states=None,
                    checkpoint: Optional[str] = None,
                    *args, **kwargs):
        if checkpoint:
            assert ".pkl" in checkpoint, \
                "checkpoint should be a .pkl file. Got {}".format(checkpoint)
            if not os.path.exists(checkpoint):
                raise RuntimeError("Checkpoint file doesn't exists.")
            with open(checkpoint, "rb") as f:
                states = pickle.load(f)

        if states:
            new_state_dict = states.get("state_dict", None)
            custom_states = states.get("custom_states", None)
            lr_scheduler_states = states.get("lr_scheduler", None)

            if new_state_dict:
                self.optimizer = serialization.from_state_dict(
                    self.optimizer, new_state_dict)
            else:
                raise RuntimeError("checkpoint has no new state_dict.")

            if custom_states:
                self._custom_states.update(custom_states)

            if lr_scheduler_states:
                if hasattr(self.lr_scheduler, "set_states_dict"):
                    self.lr_scheduler.set_states_dict(lr_scheduler_states)
                else:
                    warnings.warn(
                        "lr scheduler must have `set_states_dict` method"
                        " to support loading lr scheduler states.")
        else:
            raise RuntimeError("This checkpoint is empty."
                               "Got checkpoint {}, states {}".format(
                                   checkpoint, states))

    def get_parameters(self, cpu: bool):
        named_parameters = self.get_named_parameters(cpu)
        self.parmas_keys = list(named_parameters.keys())
        return list(named_parameters.values())

    def get_named_parameters(self, cpu: bool):
        params = self.optimizer.target
        if cpu:
            params = tree_map(lambda x: np.asarray(x), params)
        params_flat_dict = traverse_util.flatten_dict(unfreeze(params))
        return params_flat_dict

    def set_parameters(self, new_params):
        assert isinstance(new_params, dict)
        optimizer = self.optimizer
        new_params = traverse_util.unflatten_dict(new_params)

        states = serialization.to_state_dict(optimizer)
        states["target"] = new_params

        self.optimizer = serialization.from_state_dict(optimizer, states)
