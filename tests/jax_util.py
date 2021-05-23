import os

from filelock import FileLock
import numpy as np

from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Dense,
                                   Flatten, GeneralConv,
                                   MaxPool, Relu, LogSoftmax)
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp

from distml.util import override
from distml.strategy.allreduce_strategy import AllReduceStrategy
from distml.operator.jax_operator import JAXTrainingOperator

from examples.jax.jax_util.datasets import mnist, Dataloader
__all__ = ["make_jax_ar_strategy", "make_jax_ps_strategy", "ToyOperator"]


def initialization_hook(self):
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    os.environ["NCCL_SHM_DISABLE"] = "1"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


def ToyModel(num_classes):
    return stax.serial(
        GeneralConv(('HWCN', 'OIHW', 'NHWC'), 1, (3, 3), (1, 1), 'SAME'),
        BatchNorm(), Relu, AvgPool((2, 2), padding="SAME"),
        GeneralConv(('NHWC', 'HWIO', 'NHWC'), 16, (3, 3), (1, 1), 'SAME'),
        BatchNorm(), Relu,
        GeneralConv(('NHWC', 'HWIO', 'NHWC'), 32, (1, 1), (1, 1), 'SAME'),
        BatchNorm(), Relu,
        AvgPool((7, 7), padding="SAME"),
        Flatten, Dense(num_classes), LogSoftmax)


class ToyOperator(JAXTrainingOperator):
    @override(JAXTrainingOperator)
    def setup(self, config):
        batch_size = config["batch_size"]
        lr = config["lr"]

        rng_key = random.PRNGKey(0)
        input_shape = (28, 28, 1, batch_size)

        # Create model.
        init_fun, predict_fun = ToyModel(num_classes=10)

        _, init_params = init_fun(rng_key, input_shape)

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(init_params)

        with FileLock(".ray.lock"):
            train_images, train_labels, test_images, test_labels = mnist()

        if config.get("test_mode", False):
            np.random.seed(556)
            rng = np.random.randint(low=0, high=len(test_images), size=64)
            train_images = train_images[rng, ...]
            train_labels = train_labels[rng, ...]
            test_images = test_images[rng, ...]
            test_labels = test_labels[rng, ...]

        train_images = train_images.reshape(train_images.shape[0], 1, 28,
                                            28).transpose(2, 3, 1, 0)

        test_images = test_images.reshape(test_images.shape[0], 1, 28,
                                          28).transpose(2, 3, 1, 0)

        train_loader = Dataloader(
            train_images, train_labels, batch_size=batch_size, shuffle=True)
        test_loader = Dataloader(
            test_images, test_labels, batch_size=batch_size)

        def criterion(logits, targets):
            return -jnp.sum(logits * targets)

        self.register(
            model=[opt_state, init_fun, predict_fun],
            optimizer=[opt_init, opt_update, get_params],
            criterion=criterion)

        self.register_data(
            train_loader=train_loader, validation_loader=test_loader)


def make_jax_ar_strategy(world_size=2):
    strategy = AllReduceStrategy(
        training_operator_cls=ToyOperator,
        initialization_hook=initialization_hook,
        world_size=world_size,
        operator_config={
            "lr": 0.01,
            "test_mode": True,  # subset the data
            # this will be split across workers.
            "batch_size": 16
        })

    return strategy

def make_jax_ps_strategy(num_ps=2, num_worker=2):
    strategy = ParameterServerStrategy(
        training_operator_cls=ToyOperator,
        initialization_hook=initialization_hook,
        world_size=num_ps + num_worker,
        num_ps = num_ps,
        num_worker = num_worker,
        operator_config={
            "lr": 0.01,
            "test_mode": True,  # subset the data
            # this will be split across workers.
            "batch_size": 16
        })

    return strategy


# class Worker(object):
#     def __init__(self):
#         self.strategy = None
#
#     def setup_ar_strategy(self):
#         strategy = AllReduceStrategy(
#             training_operator_cls=ToyOperator,
#             initialization_hook=initialization_hook,
#             world_size=2,
#             operator_config={
#                 "lr": 0.01,
#                 "test_mode": True,  # subset the data
#                 # this will be split across workers.
#                 "batch_size": 16
#             })
#
#         self.strategy = strategy

    # def setup_ps_strategy(self):
    #     strategy = ParameterServerStrategy(
    #         training_operator_cls=ToyOperator,
    #         initialization_hook=initialization_hook,
    #         world_size=2,
    #         operator_config={
    #             "lr": 0.01,
    #             "test_mode": True,  # subset the data
    #             # this will be split across workers.
    #             "batch_size": 16
    #         })
    #
    #     self.strategy = strategy
