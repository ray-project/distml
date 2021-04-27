import numpy as np
import cupy as cp
import jax
from jax import grad, value_and_grad
import jax.numpy as jnp
from jax.lib import xla_client
from jax.dlpack import from_dlpack, to_dlpack
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure, tree_map, build_tree
from jax._src.util import unzip2
from jax.experimental.optimizers import OptimizerState

from .base_operator import TrainingOperator
from distml.strategy.util import ThroughputCollection, func_timer
from ray.util.sgd.utils import TimerCollection, AverageMeterCollection

import time 


class JAXTrainingOperator(TrainingOperator):

    def __init__(self, operator_config):
        super(JAXTrainingOperator, self).__init__(operator_config)
        # Should be set by users in the `register` function.
        self.model = None
        self.optimizer = None
        self.criterion = None
        # Models, optimizers, and criterion registered by users.
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._lr_scheduler = None

        # Data loaders for training and validation, registered by users.
        self._train_loader = None
        self._validation_loader = None

        self.setup(operator_config)

        if hasattr(operator_config, "jit_mode"):
            assert not operator_config["jit_mode"], "Not support jit in jax operator."

        self.train_step_num = 0
        print(f"device count {jax.device_count()}")

    def setup(self, *args, **kwargs):
        """Function that needs to be override by users.
        
        Example:
            # some code is the same for all users, maybe we can put it in register.
            rng_key = random.PRNGKey(0)
            input_shape = (28, 28, 1, batch_size)
            lr=0.01
            init_fun, predict_fun = ResNet18(num_classes)
            _, init_params = init_fun(rng_key, input_shape)
            
            opt_init, opt_update, get_params = optimizers.adam(lr)
            opt_state = opt_init(init_params)
            
            self.register(model=(opt_state, get_params, predict_fun), optimizer=opt_update, criterion=lambda logits, targets:-jnp.sum(logits * targets)
        
        """
        
        pass

    def register(self, 
                 *,
                 model, 
                 optimizer, 
                 criterion, 
                 lr_schedulers=None, 
                 jit_mode=False):
        """Register a few critical information about the model to operator."""
        self.criterion = criterion
        if lr_schedulers:
            self.lr_schedulers = lr_schedulers
            print("WARNING: jax not support learning rate scheduler." 
                  "This will not work.")
        
        self._register_model(model)
        self._register_optimizer(optimizer)
            
    def _register_model(self, model):
        """Re the states to a file path.

         This function shall be instantiated in framework-specific operator
         implementations.
         """
        self.opt_state = model[0]
        self.init_fun = model[1]
        self.predict_fun = model[2]

    def _register_optimizer(self, optimizer):
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
        inputs, targets = batch
        logits = self.predict_fun(params, inputs)
        return self.criterion(logits, targets)

    def derive_updates(self, batch):
        # cal_time = time.time()
        loss_val, gradient = self._calculate_gradient(self.opt_state, batch)
        # print("calculate gradient spend {:.5f}".format(time.time()-cal_time))
        
        flat_time = time.time()
        gradient_dict, tree = tree_flatten(gradient)
        assert tree == self.opt_state[1]
        # print("falt time spend {:.5f}".format(time.time()-flat_time))

        # grad_dict_time = time.time()
        if hasattr(self, "preset_keys"):
            gradient_dict = {k:g for k, g in zip(self.preset_keys, gradient_dict)}
        else:
            gradient_dict = {f"{idx}":g for idx, g in enumerate(gradient_dict)}
        # print("make a grad_dict spend {:.5f}".format(time.time()-grad_dict_time))
        return loss_val.item(), gradient_dict
        
    # @func_timer
    def apply_updates(self, gradient):
        # sort_gradient_time = time.time()
        keys, gradient = unzip2(sorted(gradient.items(), key=lambda d: int(d[0])))
        gradient = tree_unflatten(self.opt_state[1], gradient)
        # print("sorting spend {:.5f}".format(time.time()-sort_gradient_time))
        
        # opt_update_time = time.time()
        self.opt_state = self.opt_update(self.train_step_num, gradient, self.opt_state)
        self.train_step_num += 1
        # print("update time spend {:.5f}".format(time.time()-opt_update_time))

    def to_cupy(self, tensor):
        if isinstance(tensor, list):
            return list(map(self.to_cupy, tensor))
        ctensor = cp.fromDlpack(self.get_jax_dlpack(tensor))
        # assert ctensor.data.ptr == tensor.unsafe_buffer_pointer()
        return ctensor

    def to_operator_tensor(self, tensor):
        if isinstance(tensor, list):
            return list(map(self.to_operator_tensor, tensor))
        return from_dlpack(tensor.toDlpack())

    def _calculate_gradient(self, opt_state, batch):
        params = self.get_params(opt_state)
        loss_val, gradient = value_and_grad(self.loss_func)(params, batch)
        return loss_val, gradient

    def get_jax_dlpack(self, tensor):
        return xla_client._xla.buffer_to_dlpack_managed_tensor(tensor.device_buffer,
                                                               take_ownership=False)

    def validate(self, info={}):
        if not hasattr(self, "opt_state"):
            raise RuntimeError("model unset. Please register model in setup.")
        if not hasattr(self, "criterion"):
            raise RuntimeError("criterion unset. Please register criterion in setup.")
        
        params = self.get_params(self.opt_state)
        validation_loader = self._validation_loader
        metric_meters = AverageMeterCollection()

        for batch_idx, batch in enumerate(validation_loader):
            batch_info = {"batch_idx": batch_idx}
            batch_info.update(info)
            metrics = self.validate_step(params, batch, batch_info)
            metric_meters.update(metrics, n=metrics.pop("samples_num", 1))
        return metric_meters.summary()

    def validate_batch(self, batch):
        params = self.get_params(self.opt_state)
        criterion = self.criterion
        predict_fun = self.predict_fun
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
            "samples_num": samples_num
        }

    def get_parameters(self, cpu):
        params = self.get_params(self.opt_state)
        flatten_params, tree = tree_flatten(params)
        if not hasattr(self, "tree"):
            self.tree = tree

        if cpu:
            flatten_params = list(map(np.asarray, flatten_params))
        return flatten_params

    def get_named_parameters(self, cpu):
        params = self.get_parameters(cpu)
        if hasattr(self, "preset_keys"):
            dict_params = {name:p for name, p in zip(self.preset_keys, params)}
        else:
            dict_params = {f"{idx}":p for idx, p in enumerate(params)}
        return dict_params

    def set_parameters(self, new_params):
        assert isinstance(new_params, dict)

        keys, new_params = unzip2(sorted(new_params.items(), key=lambda d: int(d[0])))
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
        for idx, (subtree, new_subtree) in enumerate(zip(subtrees, new_subtrees)):
            if new_subtree != subtree:
                msg = ("input structur did not match the save params struture. "
                       "input {} and output {}.")
                raise TypeError(msg.format(subtree, new_subtree))

        self.opt_state = OptimizerState(new_states_flat, tree, new_subtrees)

    def reset_optimizer_for_params(self, params):
        keys, params = unzip2(sorted(params.items(), key=lambda d: int(d[0])))
        self.tree = tree_structure(params)
        self.opt_state = self.opt_init(params)

    def ones(self, shape, cpu=True):
        if cpu:
            return np.ones(shape)
        else:
            return jnp.ones(shape)

    def zeros(self, shape, cpu=True):
        if cpu:
            return np.zeros(shape)
        else:
            return jnp.zeros(shape)

    def ones_like(self, x, cpu=True):
        if cpu:
            return np.ones_like(x)
        else:
            return jnp.ones_like(x)

    def zeros_like(self, x, cpu=True):
        if cpu:
            return np.zeros_like(x)
        else:
            return jnp.zeros_like(x)

    def numel(self, v):
        return np.size(v)

    def asarray(self, v):
        return jnp.asarray(v)

    def clean_redundancy(self):
        del self._train_loader
        del self._validation_loader

    # TODO(HUI): use pickle to serialize parameters or states and save it.
    def save_parameters(self, checkpoint):
        raise NotImplementedError(
            "save_parameters is not support in jax operator.")

    def load_parameters(self, checkpoint):
        raise NotImplementedError(
            "load_parameters is not support in jax operator.")

    def save_states(self, states):
        raise NotImplementedError(
            "save_states is not support in jax operator.")

    def get_states(self, states):
        raise NotImplementedError(
            "get_states is not support in jax operator.")

    def load_states(self, checkpoint):
        raise NotImplementedError(
            "load_states is not support in jax operator.")

