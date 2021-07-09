import os

from filelock import FileLock
import numpy as np

import flax
from flax import optim
import flax.linen as nn

from functools import partial
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Dense, Flatten,
                                   GeneralConv, Relu, LogSoftmax)
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp

from distml.util import override
from distml.strategy.allreduce_strategy import AllReduceStrategy
from distml.strategy.ps_strategy import ParameterServerStrategy
from distml.operator.flax_operator import FLAXTrainingOperator

from examples.flax.flax_util.datasets import Dataloader
from examples.jax.jax_util.datasets import mnist

__all__ = ["make_jax_ar_strategy", "make_jax_ps_strategy", "ToyOperator"]


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    os.environ["NCCL_SHM_DISABLE"] = "1"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


class ToyModel(nn.Module):
    num_classes: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.relu = nn.relu
        self.avg_pool = partial(nn.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=self.num_classes)
        self.log_softmax = nn.log_softmax

    @nn.compact
    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.fc1(x)
        x = self.log_softmax(x)
        return x


class ToyOperator(FLAXTrainingOperator):
    @override(FLAXTrainingOperator)
    def setup(self, config):
        batch_size = config["batch_size"]
        lr = config["lr"]

        key1, key2 = random.split(random.PRNGKey(0))
        input_shape = (batch_size, 28, 28, 1)

        model = ToyModel(num_classes=10)
        x = random.normal(key1, input_shape)
        params = model.init(key2, x)

        optimizer_def = optim.Adam(learning_rate=lr) # Choose the method
        optimizer = optimizer_def.create(params) # Create the wrapping optimizer with initial parameters
        
        with FileLock(".ray.lock"):
            train_images, train_labels, test_images, test_labels = mnist()

        if config.get("test_mode", False):
            np.random.seed(556)
            rng = np.random.randint(low=0, high=len(test_images), size=64)
            train_images = train_images[rng, ...]
            train_labels = train_labels[rng, ...]
            test_images = test_images[rng, ...]
            test_labels = test_labels[rng, ...]
            
        train_images = train_images.reshape(
            train_images.shape[0], 1, 28, 28).transpose(0, 2, 3, 1)
        test_images = test_images.reshape(
            test_images.shape[0], 1, 28, 28).transpose(0, 2, 3, 1)

        train_loader = Dataloader(
            train_images, train_labels, batch_size=batch_size, shuffle=True)
        test_loader = Dataloader(
            test_images, test_labels, batch_size=batch_size)

        def criterion(logits, targets):
            return -jnp.sum(logits * targets)

        self.register(model=model, optimizer=optimizer, criterion=criterion)
    
        self.register_data(train_loader=train_loader, validation_loader=test_loader)


def make_flax_ar_strategy(world_size=2, backend="nccl", group_name="default"):
    strategy = AllReduceStrategy(
        training_operator_cls=ToyOperator,
        initialization_hook=initialization_hook,
        world_size=world_size,
        backend=backend,
        group_name=group_name,
        operator_config={
            "lr": 0.01,
            "test_mode": True,  # subset the data
            # this will be split across workers.
            "batch_size": 16
        })

    return strategy


def make_flax_ps_strategy(num_ps=2, num_worker=2):
    strategy = ParameterServerStrategy(
        training_operator_cls=ToyOperator,
        initialization_hook=initialization_hook,
        world_size=num_ps + num_worker,
        num_ps=num_ps,
        num_worker=num_worker,
        operator_config={
            "lr": 0.01,
            "test_mode": True,  # subset the data
            # this will be split across workers.
            "batch_size": 16
        })

    return strategy
