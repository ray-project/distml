import os
import argparse

from filelock import FileLock

import ray
from distml.operator.jax_operator import JAXTrainingOperator
from distml.strategy.allreduce_strategy import AllReduceStrategy

from ray.util.sgd.utils import override

from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp
from jax_util.resnet import ResNet18, ResNet50, ResNet101
from jax_util.datasets import mnist, Dataloader


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


class MnistTrainingOperator(JAXTrainingOperator):
    @override(JAXTrainingOperator)
    def setup(self, config):
        batch_size = config["batch_size"]
        rng_key = random.PRNGKey(0)
        input_shape = (28, 28, 1, batch_size)
        lr = config["lr"]
        model_name = config["model_name"]
        num_classes = config["num_classes"]

        if model_name == "resnet18":
            init_fun, predict_fun = ResNet18(num_classes)
        elif model_name == "resnet50":
            init_fun, predict_fun = ResNet50(num_classes)
        elif model_name == "resnet101":
            init_fun, predict_fun = ResNet101(num_classes)
        else:
            raise RuntimeError("Unrecognized model name")

        _, init_params = init_fun(rng_key, input_shape)

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(init_params)

        with FileLock(".ray.lock"):
            train_images, train_labels, test_images, test_labels = mnist()

        train_images = train_images.reshape(train_images.shape[0], 1, 28,
                                            28).transpose(2, 3, 1, 0)
        test_images = test_images.reshape(test_images.shape[0], 1, 28,
                                          28).transpose(2, 3, 1, 0)

        train_loader = Dataloader(
            train_images, train_labels, batch_size=batch_size, shuffle=True)
        test_loader = Dataloader(
            test_images, test_labels, batch_size=batch_size)

        self.register(
            model=[opt_state, init_fun, predict_fun],
            optimizer=[opt_init, opt_update, get_params],
            criterion=lambda logits, targets: -jnp.sum(logits * targets))

        self.register_data(
            train_loader=train_loader, validation_loader=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for connecting to the Ray cluster")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of epochs to train.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enables FP16 training with apex. Requires `use-gpu`.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        help="model, Optional: resnet18, resnet50, resnet101.")

    args, _ = parser.parse_known_args()

    if args.address:
        ray.init(args.address)
    else:
        ray.init(
            num_gpus=args.num_workers,
            num_cpus=args.num_workers * 2,
            log_to_driver=True)

    strategy = AllReduceStrategy(
        training_operator_cls=MnistTrainingOperator,
        world_size=args.num_workers,
        operator_config={
            "lr": 0.01,
            "batch_size": 128,
            "num_workers": args.num_workers,
            "num_classes": 10,
            "model_name": args.model_name
        })

    for i in range(args.num_epochs):
        strategy.train()
    print(strategy.validate())

    strategy.shutdown()
    print("success!")
