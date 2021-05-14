from abc import ABCMeta
from abc import abstractmethod
import logging
from typing import AbstractSet, Callable, Any, Mapping, Optional

import ray

logger = logging.getLogger(__name__)


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self,
                 *,
                 training_operator_cls,
                 operator_config: Optional[Mapping[str, Any]] = None,
                 initialization_hook: Optional[Callable] = None,
                 world_size: int = 2,
                 backend: str = "nccl",
                 group_name: str = "default",
                 num_cpus_per_worker: int = 1,
                 num_gpus_per_worker: int = 1,
                 **kwargs):
        self.training_operator_cls = training_operator_cls
        self.initialization_hook = initialization_hook
        if world_size < 2:
            raise RuntimeError(
                "ray.util.distml does not support single-process training "
                "at this moment.")
        self.world_size = world_size
        self.backend = backend
        self.group_name = group_name
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self._operator_config = {} if not operator_config \
            else operator_config
        if not ray.is_initialized() and self.world_size > 1:
            logger.info("Automatically initializing single-node Ray. To use "
                        "multi-node training, be sure to run `ray.init("
                        "address='auto')` before instantiating the Strategy.")
            ray.init()
        self._start_workers()

    @abstractmethod
    def train(self, *args, **kwargs):
        """Run the training on parallel workers."""
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        """Call operator validate to evaluate val_dataloader.
        """
        raise NotImplementedError()

    @abstractmethod
    def save_parameters(self, checkpoint: str):
        """Saves the Trainer state to the provided checkpoint path.

        Args:
            checkpoint (str): Path to target checkpoint file.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_parameters(self, checkpoint: str):
        """Loads the Trainer state to the provided checkpoint path.

        Args:
            checkpoint (str): Path to target checkpoint file.
        """
        raise NotImplementedError()

    @abstractmethod
    def _start_workers(self):
        """Start all the workers to be used for training."""
        raise NotImplementedError()

    @abstractmethod
    def _init_strategy(self):
        """Strategy-specific prep-up."""
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self, force: bool = False):
        """Kill all workers."""
        raise NotImplementedError()


class BaseDataParallelGroup:
    """Spawn a actor group for data-parallel training."""

    def __init__(self,
                 actor_params: Mapping[str, Any],
                 dist_params: Mapping[str, Any],
                 num_cpus_per_actor: int,
                 num_gpus_per_actor: int,
                 initialization_hook: Optional[Callable],
                 **kwargs):
        self._actor_params = actor_params
        self._dist_params = dist_params
        self._backend = self._dist_params["backend"]
        self._group_name = self._dist_params["group_name"]
        self._num_cpus_per_actor = num_cpus_per_actor
        self._num_gpus_per_actor = num_gpus_per_actor
        self._initialization_hook = initialization_hook

        # try to unroll the dist_params
        self._backend = self._dist_params["backend"]
        self._group_name = self._dist_params["group_name"]

    @property
    def world_size(self):
        return len(self._actors)

    @property
    def backend(self):
        return self._backend

    @property
    def group_name(self):
        return self._group_name

    @abstractmethod
    def _setup_collective_group(self, *args, **kwargs):
        """All actors setup operators."""
        raise NotImplementedError()

    @abstractmethod
    def setup_operator(self):
        """All actors setup operators."""
        raise NotImplementedError()

    @abstractmethod
    def _start_actors(self, num_actors):
        """Start all actors."""
        raise NotImplementedError()

    @abstractmethod
    def make_iterator(self, training: bool = True):
        """Make iterator."""
        raise NotImplementedError()

    @abstractmethod
    def get_data_loader_len(self, training: bool = True):
        """Return the number of batches in the data loader."""
        raise NotImplementedError()

    @abstractmethod
    def validate_batch(self):
        """Validate one batch and return batch metrics."""
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self, force: bool = False):
        """Shutdown all actors."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset group."""
        raise NotImplementedError()

    @abstractmethod
    def save_parameters(self, checkpoint: str):
        """Let the first actor save parameters."""
        raise NotImplementedError()

    @abstractmethod
    def load_parameters(self, checkpoint: str):
        """All actor load parameters from checkpoint."""
        raise NotImplementedError()

    @abstractmethod
    def set_parameters(self, params):
        """Input params and replace the model parameters."""
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self, cpu: bool = False):
        """Return parameters from the first actor."""
        raise NotImplementedError()

    @abstractmethod
    def get_named_parameters(self, cpu: bool = False):
        """Return named parameters from the first actor."""
        raise NotImplementedError()
