import os
import argparse

from filelock import FileLock

from tqdm import trange

import ray
from distml.operator.flax_operator import FLAXTrainingOperator
from distml.strategy.allreduce_strategy import AllReduceStrategy
from distml.strategy.ps_strategy import ParameterServerStrategy
from ray.util.sgd.utils import BATCH_SIZE, override

import numpy as np
import numpy.random as npr

import jax
from jax import jit, grad, random
from jax.tree_util import tree_flatten
import jax.numpy as jnp

import flax 
import flax.optim as optim

from flax_util.models import ToyModel
from flax_util.datasets import Dataloader
from examples.jax.jax_util.datasets import mnist


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"

       
class MnistTrainingOperator(FLAXTrainingOperator):
    @override(FLAXTrainingOperator)
    def setup(self, config):
        key1, key2 = random.split(random.PRNGKey(0))
        batch_size = config["batch_size"]
        input_shape = (batch_size, 28, 28, 1)
        lr = config["lr"]

        model = ToyModel(num_classes=10)
        x = random.normal(key1, input_shape)
        params = model.init(key2, x)

        optimizer_def = optim.Adam(learning_rate=lr) # Choose the method
        optimizer = optimizer_def.create(params) # Create the wrapping optimizer with initial parameters
        
        with FileLock(".ray.lock"):
            train_images, train_labels, test_images, test_labels = mnist()

        if config.get("test_mode", False):
            np.random.seed(556)
            rng = np.random.randint(low=0, high=len(test_images), size=1000)
            train_images = train_images[rng, ...]
            train_labels = train_labels[rng, ...]
            test_images = test_images[rng, ...]
            test_labels = test_labels[rng, ...]

        train_images = train_images.reshape(train_images.shape[0], 1, 28, 28).transpose(0, 2, 3, 1)
        test_images = test_images.reshape(test_images.shape[0], 1, 28, 28).transpose(0, 2, 3, 1)

        train_loader = Dataloader(train_images, train_labels, batch_size=batch_size, shuffle=True)
        test_loader = Dataloader(test_images, test_labels, batch_size=batch_size)
        
        self.register(model=model, optimizer=optimizer, criterion=lambda logits, targets:-jnp.sum(logits * targets))
    
        self.register_data(train_loader=train_loader, validation_loader=test_loader)


def make_ar_strategy(args):
    strategy = AllReduceStrategy(
        training_operator_cls=MnistTrainingOperator,
        world_size=args.num_worker,
        operator_config={
            "lr": 0.01,
            "test_mode": args.smoke_test,  # subset the data
                # this will be split across workers.
            "batch_size": 128,
            "num_classes": 10,
        },
        initialization_hook=initialization_hook)

    return strategy


def make_ps_strategy(args):
    strategy = ParameterServerStrategy(
        training_operator_cls=MnistTrainingOperator,
        world_size=args.num_worker,
        num_worker=args.num_worker - args.num_ps,
        num_ps=args.num_ps,
        operator_config={
            "lr": 0.01,
            "test_mode": args.smoke_test,  # subset the data
                # this will be split across workers.
            "batch_size": 128,
            "num_classes": 10,
        },
        initialization_hook=initialization_hook)

    return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for connecting to the Ray cluster")
    parser.add_argument(
        "--num-worker",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--num-ps",
        type=int,
        default=1,
        help="Sets number of servers for training. Only for ps_strategy.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enables FP16 training with apex. Requires `use-gpu`.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.")
    parser.add_argument(
        "--tune", action="store_true", default=False, help="Tune training")
    parser.add_argument(
        "--strategy", type=str, default="ar", help="strategy type, Optional: ar, ps")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,6,7"

    args, _ = parser.parse_known_args()
    ray.init(num_gpus=args.num_worker, num_cpus=args.num_worker, log_to_driver=True)

    if args.strategy == "ar":
        strategy = make_ar_strategy(args)
    elif args.strategy == "ps":
        strategy = make_ps_strategy(args)
    else:
        raise RuntimeError("Unrecognized strategy type. Except 'ar' or 'ps'"
                           "Got {}".format(args.strategy))

    for i in range(args.num_epochs):
        strategy.train()
    print(strategy.validate())
    strategy.shutdown()
    print("success!")

