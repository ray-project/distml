from abc import ABCMeta
from abc import abstractmethod
import logging

import ray

logger = logging.getLogger(__name__)

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, 
                 *,
                 training_operator_cls, 
                 operator_config=None,
                 initialization_hook=None,
                 world_size=2,
                 num_cpus_per_worker=1,
                 num_gpus_per_worker=1,
                 **kwargs):
        self.training_operator_cls = training_operator_cls
        self.initialization_hook = initialization_hook
        if world_size < 2:
            raise RuntimeError("ray.util.distml does not support single-process training "
                               "at this moment.")
        self.world_size = world_size
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_gpus_per_worker = num_gpus_per_worker
        self._operator_config = {} if operator_config is None else operator_config

        if not ray.is_initialized() and self.max_replicas > 1:
            logger.info("Automatically initializing single-node Ray. To use "
                        "multi-node training, be sure to run `ray.init("
                        "address='auto')` before instantiating the Trainer.")
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
    def save_parameters(self, checkpoint):
        """Saves the Trainer state to the provided checkpoint path.
        Args:
            checkpoint (str): Path to target checkpoint file.
        """
        raise NotImplementedError()

    @abstractmethod
    def load_parameters(self, checkpoint):
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
    def shutdown(self, force=False):
        """Kill all workers.
        """
        raise NotImplementedError()