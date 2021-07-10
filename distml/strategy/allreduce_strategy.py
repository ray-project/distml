import logging
from typing import List, Callable, Mapping, Any, Optional, Dict

import ray
import ray.util.collective as col
from ray.util.sgd.utils import AverageMeterCollection

import numpy as np

from distml.strategy.base_strategy import BaseStrategy, BaseDataParallelGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AllReduceStrategy(BaseStrategy):
    """Strategy that trains a model via collective AllReduce.

    Args:
        training_operator_cls (TrainingOperator):
            Custom training operator class.
        operator_config (dict): operator config specified by users.
        initialization_hook (function): A function to call on all training
            workers when they are first initialized. This could be useful to
            set environment variables for all the worker processes.
        world_size (int): The number of parallel workers.
        num_cpus_per_worker (int): number of CPUs allocated per worker.
        num_gpus_per_worker (int): number of GPUs allocated per worker.
    """

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
        super(AllReduceStrategy, self). \
            __init__(training_operator_cls=training_operator_cls,
                     operator_config=operator_config,
                     initialization_hook=initialization_hook,
                     world_size=world_size,
                     backend=backend,
                     group_name=group_name,
                     num_cpus_per_worker=num_cpus_per_worker,
                     num_gpus_per_worker=num_gpus_per_worker,
                     **kwargs)
        self._global_batch_size = None

        if operator_config and operator_config.get("batch_size"):
            self._global_batch_size = operator_config.get("batch_size")

        self._init_strategy()

    def train(self, num_steps: Optional[int] = None) -> Dict:
        """Run the training on parallel workers.

        Args:
            num_steps (int): number of steps to train. If none, the
                function will simply train for one epoch.

        Returns:
            metric (dict): metric of the training set.
        """
        # TODO(HUI): metric use hook to control.

        # TODO (Hao): add fault tolerance using `max_retries`.
        steps = num_steps if num_steps \
            else self.data_parallel_group.get_data_loader_len()

        # TODO(Hao): this call should be hidden inside Replica.
        self.data_parallel_group.make_iterator()
        for idx in range(steps):
            metric = self.data_parallel_group.train_batch()
            print("Step: {}/{}".format(idx, steps))
        return metric

    def validate(self, num_steps: Optional[int] = None) -> Dict:
        """Evaluates the model on the validation data.

        Args:
            num_steps (int): number of batches to evaluate. If None, the
                function will simply validate across the entire validation
                dataset.

        Returns:
            metric (dict): metric of the validate set.
        """
        steps = num_steps if num_steps \
            else self.data_parallel_group.get_data_loader_len(training=False)

        metrics = [
            AverageMeterCollection()
            for _ in range(len(self.data_parallel_group.replicas))
        ]

        self.data_parallel_group.make_iterator(training=False)
        for idx in range(steps):
            batch_metrics = self.data_parallel_group.validate_batch()

            for metric_idx, metric in enumerate(batch_metrics):
                num_sample = metric.pop("num_sample")
                metrics[metric_idx].update(metric, n=num_sample)

        # TODO: validate result should be the same in all workers
        return metrics[0].summary()

    def _init_strategy(self):
        """Do initialization for the distributed strategy."""
        # All sync with replica 0
        init_weights = self.data_parallel_group.get_named_parameters(cpu=True)
        # all replicas get synced
        self.data_parallel_group.set_parameters(init_weights)

    def _start_workers(self):
        """Create distributed workers on the Ray cluster for distributed training.

        Specifically, this function will spawn the necessary actor processes
        depending on the strategy used, and arrange and pass their required
        arguments.
        """
        # TODO (Hao): infer the per-replica batch size here...
        # so here we get multiple sets of params that will be passed around:
        # (1) Those for setting up replica
        operator_config = self._operator_config.copy()
        replica_params = dict(
            training_operator_cls=self.training_operator_cls,
            operator_config=operator_config)
        # (2) params for setting up collective group and strategy prep-ups.
        dist_params = dict(
            strategy="allreduce",
            backend=self.backend,
            group_name=self.group_name,
        )
        group_init_args = dict(
            actor_params=replica_params,
            dist_params=dist_params,
            initialization_hook=self.initialization_hook,
            num_cpus_per_actor=self.num_cpus_per_worker,
            num_gpus_per_actor=self.num_gpus_per_worker)
        self.data_parallel_group = DataParallelGroup(**group_init_args)
        # Once the group is created, we start it.
        self.data_parallel_group.start_replicas(self.world_size)

    def shutdown(self, force: bool = False):
        self.data_parallel_group.shutdown(force=force)

    def get_states(self):
        return self.data_parallel_group.get_states()

    def save_states(self, checkpoint: str):
        self.data_parallel_group.save_states(checkpoint)

    def load_states(self, states=None, checkpoint: Optional[str] = None):
        self.data_parallel_group.load_states(states, checkpoint)


class Replica:
    """Express the training semantics of a data-parallel replica.

    This class includes some glue code between the user-provided operator
    and Ray collective group setup.
    """

    def __init__(self, training_operator_cls,
                 operator_config: Optional[Mapping[str, Any]]):
        self.training_operator_cls = training_operator_cls
        self.operator_config = operator_config
        # Training operator
        self.training_operator = None

        # collective-related information
        self._world_size = None
        self._rank = None
        self._group_name = None

        # Iterators
        self.train_iterator = None
        self.validation_iterator = None

    def setup_operator(self):
        """Instantiate the training operator."""
        self.training_operator = self.training_operator_cls(
            operator_config=self.operator_config)

    def setup_collective_group(self,
                               rank: int,
                               world_size: str,
                               backend: str,
                               group_name: str = "default"):
        self._rank = rank
        self._group_name = group_name
        self._world_size = world_size
        col.init_collective_group(
            world_size, rank, backend=backend, group_name=group_name)

    def make_iterator(self, training: bool = True):
        """Convert loader to be an iterator at the start of an epoch."""
        # TODO(Hao): need to check whether reaching the boundary of iterator
        #            instead of making a new one every time.
        if training:
            self.train_iterator = iter(self.train_loader)
        else:
            self.validation_iterator = iter(self.validation_loader)

    def get_data_loader_len(self, training: bool = True) -> int:
        """Return the number of batches in the data loader."""
        loader = self.train_loader if training \
            else self.validation_loader
        if hasattr(loader, "__len__"):
            return len(loader)
        else:
            raise RuntimeError(
                "Data loader has no attribute `__len__`. "
                "Please set `num_steps` in `train()` or `validate()`.")

    def train_batch(self) -> Dict:
        metrics = {}
        try:
            batch = next(self.train_iterator)
        except StopIteration or NameError:
            self.make_iterator()
            batch = next(self.train_iterator)
        loss_val, updates = self.derive_updates(batch)
        assert isinstance(updates, dict)

        metrics["train_loss"] = loss_val
        for _, g in updates.items():
            cg = self.training_operator.to_cupy(g)
            col.allreduce(cg, self.group_name)
            cg = cg / float(self.world_size)
        self.apply_updates(updates)
        return metrics

    def derive_updates(self, batch) -> Dict:
        return self.training_operator.derive_updates(batch)

    def apply_updates(self, updates):
        self.training_operator.apply_updates(updates)

    def updates_transform(self, updates):
        return self.training_operator.updates_transform(updates)

    def validate_batch(self):
        try:
            batch = next(self.validation_iterator)
        except StopIteration or NameError:
            self.make_iterator(training=False)
            batch = next(self.validation_iterator)
        batch_metric = self.training_operator.validate_batch(batch)
        return batch_metric

    def shutdown(self):
        # destroy the collective group resources on this process
        col.destroy_collective_group(self.group_name)
        if self.training_operator:
            del self.training_operator
        return 1

    def get_states(self):
        return self.training_operator.get_states()

    def save_states(self, checkpoint: str):
        self.training_operator.save_states(checkpoint)

    def load_states(self, states=None, checkpoint: Optional[str] = None):
        self.training_operator.load_states(states, checkpoint)

    def get_parameters(self, cpu: bool) -> List:
        return self.training_operator.get_parameters(cpu)

    def get_named_parameters(self, cpu: bool) -> Dict:
        return self.training_operator.get_named_parameters(cpu)

    def set_parameters(self, params):
        self.training_operator.set_parameters(params)

    def apply(self, fn: Callable):
        """Apply a function in the replica process."""
        return fn()

    @property
    def train_loader(self):
        return self.training_operator._get_train_loader()

    @property
    def validation_loader(self):
        return self.training_operator._get_validation_loader()

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @property
    def group_name(self):
        return self._group_name


class DataParallelGroup(BaseDataParallelGroup):
    """Spawn a replica group for data-parallel training."""

    def __init__(self, actor_params: Mapping[str, Any],
                 dist_params: Mapping[str, Any], num_cpus_per_actor: int,
                 num_gpus_per_actor: int,
                 initialization_hook: Optional[Callable]):
        super(DataParallelGroup, self).__init__(
            actor_params=actor_params,
            dist_params=dist_params,
            num_cpus_per_actor=num_cpus_per_actor,
            num_gpus_per_actor=num_gpus_per_actor,
            initialization_hook=initialization_hook)

    @property
    def _replica_params(self):
        return self._actor_params

    @property
    def replicas(self):
        return self._actors

    @property
    def group_name(self):
        return self._group_name

    def start_replicas(self, num_replicas: int):
        self._start_actors(num_replicas)

    def _start_actors(self, num_replicas: int):
        assert num_replicas > 1
        RemoteReplica = ray.remote(
            num_cpus=self._num_cpus_per_actor,
            num_gpus=self._num_gpus_per_actor)(Replica)
        self._actors = [
            RemoteReplica.remote(**self._replica_params)
            for _ in range(num_replicas)
        ]

        # apply init_hook
        if self._initialization_hook:
            self.apply_all_replicas(self._initialization_hook)

        # setup the rank and group in each replica
        group_setup_refs = self._setup_collective_group(
            self.world_size, self.backend, self.group_name)
        ray.get(group_setup_refs)

        # setup the model training operator
        operator_setups = self._setup_operator()
        ray.get(operator_setups)

    def _make_iterator(self, training: bool):
        return [
            replica.make_iterator.remote(training=training)
            for replica in self.replicas
        ]

    def make_iterator(self, training: bool = True):
        ray.get(self._make_iterator(training=training))

    def get_data_loader_len(self, training: bool = True):
        """Return the number of batches in the data loader."""
        lens = ray.get([
            replica.get_data_loader_len.remote(training=training)
            for replica in self.replicas
        ])
        if len(set(lens)) != 1:
            # TODO(Hao): is this correct after we add distributed data loader?
            raise RuntimeError(
                "All replica should have the same dataloader len.")
        return lens[0]

    def train_batch(self):
        metrics = {}
        loss_vals = ray.get(
            [replica.train_batch.remote() for replica in self.replicas])
        train_loss_list = [d["train_loss"] for d in loss_vals]
        metrics["train_loss"] = np.mean(train_loss_list)
        return metrics

    def validate_batch(self):
        rets = [replica.validate_batch.remote() for replica in self.replicas]
        stats = ray.get(rets)
        return stats

    def shutdown(self, force: bool = False):
        rets = [replica.shutdown.remote() for replica in self.replicas]
        stats = ray.get(rets)
        return stats

    def reset(self):
        pass

    def get_states(self):
        rets = [self.replicas[0].get_states.remote()]
        return ray.get(rets)[0]

    def save_states(self, checkpoint: str):
        rets = [self.replicas[0].save_states.remote(checkpoint)]
        ray.get(rets)

    def load_states(self, states=None, checkpoint: Optional[str] = None):
        rets = [
            replica.load_states.remote(states, checkpoint)
            for replica in self.replicas
        ]
        ray.get(rets)

    def set_parameters(self, params):
        rets = [
            replica.set_parameters.remote(params)
            for _, replica in enumerate(self.replicas)
        ]
        ray.get(rets)

    def get_parameters(self, cpu: bool = False):
        ret = self.replicas[0].get_parameters.remote(cpu)
        return ray.get(ret)[0]

    def get_named_parameters(self, cpu: bool = False):
        ret = self.replicas[0].get_named_parameters.remote(cpu)
        return ray.get([ret])[0]

    def apply_all_replicas(self, fn: Callable):
        """Apply fn in all replica processes and wait until completion."""
        return ray.get(self._apply_all_replicas(fn))

    def _apply_all_replicas(self, fn):
        """Apply a function fn in all replica processes."""
        return [replica.apply.remote(fn) for replica in self.replicas]

    def _setup_collective_group(self,
                                group_size: int,
                                backend: int,
                                group_name: str = "default"):
        refs = [
            replica.setup_collective_group.remote(
                rank=i,
                world_size=group_size,
                backend=backend,
                group_name=group_name)
            for i, replica in enumerate(self.replicas)
        ]
        return refs

    def _setup_operator(self):
        refs = [
            replica.setup_operator.remote()
            for i, replica in enumerate(self.replicas)
        ]
        return refs
