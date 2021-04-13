# TODO(Hao): could make an interface class and use factory pattern to set up strategies/trainers.
import numpy as np
import ray
import ray.util.collective as col
import ray.util.collective.types as types
from ray.util.distml.base_trainer import BaseTrainer

tqdm = None
try:
    from tqdm import tqdm
except ImportError:
    pass


class AllReduceStrategy(BaseTrainer):
    def __init__(self,
                 *,
                 training_operator_cls,
                 operator_config=None,
                 initialization_hook=None,
                 world_size=2,
                 num_cpus_per_worker=1,
                 num_gpus_per_worker=1,
                 use_tqdm=True,
                 **kwargs):
        super(AllReduceStrategy, self).\
            __init__(training_operator_cls=training_operator_cls,
                     operator_config=operator_config,
                     initialization_hook=initialization_hook,
                     world_size=world_size,
                     num_cpus_per_worker=num_cpus_per_worker,
                     num_gpus_per_worker=num_gpus_per_worker,
                     use_tqdm=use_tqdm,
                     **kwargs)

    def train(self, *, max_retries=1, info={}):
        if self._use_tqdm:
            desc = ""
            if "epoch_idx" in info.keys():
                if "num_epochs" in info.keys():
                    desc = f"Epoch {info['epoch_idx'] + 1}/{info['num_epochs']} "
                else:
                    desc = f"Epoch {info['epoch_idx'] + 1} "

            total = self.data_parallel_group.get_train_loader_len()
            _progress_bar = tqdm(
                total=total, desc=desc, unit="batch", leave=False)
            postfix = {}
            if "val_acc" in info.keys():
                postfix.update(val_acc=info["val_acc"])

        self.data_parallel_group.make_iterator()
        for idx in range(self.data_parallel_group.get_train_loader_len()):
            metrics = self.data_parallel_group.train_batch()
            if self._use_tqdm:
                _progress_bar.n = idx + 1
                if "train_loss" in metrics:
                    postfix.update(loss=metrics["train_loss"])
                _progress_bar.set_postfix(postfix)
        return info

    def validate(self, *info):
        stats = self.data_parallel_group.validate(info=info)
        return stats[0] # validate result should be the same in all workers

    def _start_workers(self):
        """Create distributed workers on the Ray cluster for distributed training.

        Specifically, this function will spawn the necessary actor processes depending
        on the strategy used, and arrange and pass their required arguments.
        """
        # TODO (Hao): infer the per-replica batch size here...

        # so here we get multiple sets of params that will be passed around:
        # (1) Those for setting up replica
        operator_config = self._operator_config.copy()
        replica_params = dict(
            training_operator_cls = self.training_operator_cls,
            operator_config = operator_config
        )
        # (2) params for setting up collective group and the strategy-related prep-ups
        dist_params = dict(
            strategy="allreduce",
            backend="nccl",
            group_name="default",
        )
        group_init_args = dict(
            replica_params=replica_params,
            dist_params=dist_params,
            initialization_hook=self.initialization_hook,
            num_cpus_per_worker=self.num_cpus_per_worker,
            num_gpus_per_worker=self.num_gpus_per_worker
        )
        self.data_parallel_group = DataParallelGroup(**group_init_args)
        # Once the group is created, we start it.
        self.data_parallel_group.start_replicas(self.world_size)

    def shutdown(self, force=False):
        self.data_parallel_group.shutdown(force=force)

    def save_parameters(self, checkpoint):
        self.data_parallel_group.save_parameters(checkpoint)

    def load_parameters(self, checkpoint):
        self.data_parallel_group.load_parameters(checkpoint)


class Replica:
    """Express the training semantics of a data-parallel replica.

    This class includes some glue code between the user-provided operator
    and Ray collective group setup.
    """
    def __init__(self,
                 training_operator_cls,
                 operator_config):
        self.training_operator_cls = training_operator_cls
        self.operator_config = operator_config

        if "use_tqdm" in operator_config.keys():
            self._use_tqdm = operator_config["use_tqdm"]
        else:
            self._use_tqdm = False

        if tqdm is None and self._use_tqdm:
            raise ValueError("tqdm must be installed to use tqdm in training.")

        # Training operator
        self.training_operator = None

        # collective-related information
        self._world_size = None
        self._rank = None
        self._group_name = None

        # Training iterators
        self.train_iterator = None

    def setup_operator(self):
        """Instantiate the training operator."""
        self.training_operator = self.training_operator_cls(
            operator_config=self.operator_config)

    def setup_collective_group(self, rank, world_size, backend, group_name="default"):
        self._rank = rank
        self._group_name = group_name
        self._world_size = world_size
        col.init_collective_group(world_size, rank,
                                  backend=backend,
                                  group_name=group_name)

    def make_iterator(self):
        """Convert loader to be an iterator at the start of an epoch."""
        self.iterator = iter(self.training_operator._get_train_loader())

    def start_iteration(self):
        self.iterator = iter(self.training_operator._get_train_loader())

    def get_train_loader_len(self):
        return len(self.training_operator._get_train_loader())

    def train_batch(self):
        metrics = {}
        try:
            batch = next(self.iterator)
        except StopIteration and NameError:
            raise RuntimeError(
                "iterator has ran out. Please use `start_iteration` to update iterator")
        
        # loss_val should be in cpu, this convertion should be done in operator.
        loss_val, updates = self.derive_updates(batch)
        metrics["train_loss"] = loss_val
        for _, g in updates.items():
            cg = self.training_operator.to_cupy(g)
            # in-place allreduce
            col.allreduce(cg)
            cg = cg / float(self.world_size)
        self.apply_updates(updates)
        return metrics

    def derive_updates(self, batch):
        # TODO (Hao): handling data loader next.
        # TODO (Hao): change it to derive_update and apply_update.
        return self.training_operator.derive_updates(batch)

    def apply_updates(self, updates):
        self.training_operator.apply_updates(updates)

    def updates_transform(self, updates):
        return self.training_operator.updates_transform(updates)

    def validate(self, info={}):
        return self.training_operator.validate(info)

    def shutdown(self):
        # destroy the collective group resources on this process
        col.destroy_collective_group(self.group_name)
        if self.training_operator:
            del self.training_operator
        return 1

    def save_parameters(self, checkpoint):
        self.training_operator.save_parameters(checkpoint)

    def load_parameters(self, checkpoint):
        self.training_operator.load_parameters(checkpoint)

    def apply(self, fn):
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


class DataParallelGroup:
    """Spawn a group a replicas for data-parallel training."""
    def __init__(self,
                 replica_params,
                 dist_params,
                 initialization_hook,
                 num_cpus_per_worker,
                 num_gpus_per_worker):
        self._replica_params = replica_params
        self._dist_params = dist_params

        # try to unroll the dist_params
        self._backend = self._dist_params["backend"]
        self._group_name = self._dist_params["group_name"]

        self._initialization_hook = initialization_hook
        self._num_cpus_per_worker = num_cpus_per_worker
        self._num_gpus_per_worker = num_gpus_per_worker
        self._replicas = None

    @property
    def replicas(self):
        return self._replicas

    @property
    def world_size(self):
        return len(self._replicas)

    @property
    def backend(self):
        return self._backend

    @property
    def group_name(self):
        return self._group_name

    def start_replicas(self, num_replicas):
        assert num_replicas > 1
        RemoteReplica = ray.remote(num_cpus=self._num_cpus_per_worker,
                                   num_gpus=self._num_gpus_per_worker)(Replica)

        self._replicas = [
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

    def _make_iterator(self):
        return [replica.make_iterator.remote()
                for replica in self.replicas]

    def make_iterator(self):
        ray.get(self._make_iterator())

    def start_iteration(self):
        rets = [replica.start_iteration.remote() 
                for replica in self.replicas]

    def get_train_loader_len(self):
        lens =  ray.get([replica.get_train_loader_len.remote()
                         for replica in self.replicas])
        if len(set(lens)) != 1:
            raise RuntimeError("All actors should have the same dataloader len.")
        return lens[0]

    def train_batch(self):
        metrics = {}
        loss_vals = ray.get([replica.train_batch.remote()
                             for _, replica in enumerate(self.replicas)])
        train_loss_list = [d["train_loss"] for d in loss_vals]
        metrics["train_loss"] = np.mean(train_loss_list)

        return metrics

    def validate(self, info={}):
        rets = [replica.validate.remote(info=info)
                for _, replica in enumerate(self.replicas)]
        stats = ray.get(rets)
        return stats

    def shutdown(self, force=False):
        rets = [replica.shutdown.remote()
                for _, replica in enumerate(self.replicas)]
        stats = ray.get(rets)
        return stats

    def reset(self):
        pass

    def save_parameters(self, checkpoint):
        rets = [self.replicas[0].save_parameters.remote(checkpoint)]
        ray.get(rets)

    def load_parameters(self, checkpoint):
        rets = [replica.load_parameters.remote(checkpoint)
                for _, replica in enumerate(self.replicas)]
        ray.get(rets)

    def set_parameters(self, params):
        rets = [replica.set_parameters.remote(params)
                for _, replica in enumerate(self.replicas)]
        ray.get(rets)
        
    def get_parameters(self, cpu=False):
        ret = self.replicas[0].get_parameters.remote(cpu)
        return ray.get(ret)[0]

    def get_named_parameters(self, cpu=False):
        ret = self.replicas[0].get_named_parameters.remote(cpu)
        return ray.get([ret])[0]

    def apply_all_replicas(self, fn):
        """Apply a function fn in all replica processes and wait for their completion."""
        return ray.get(self._apply_all_replicas(fn))

    def _apply_all_replicas(self, fn):
        """Apply a function fn in all replica processes."""
        return [replica.apply.remote(fn) for replica in self.replicas]

    def _setup_collective_group(self, world_size, backend, group_name="default"):
        refs = [
            replica.setup_collective_group.remote(
                rank=i,
                world_size=world_size,
                backend=backend,
                group_name=group_name)
                for i, replica in enumerate(self.replicas)]
        return refs

    def _setup_operator(self):
        refs = [replica.setup_operator.remote()
                  for i, replica in enumerate(self.replicas)]
        return refs
