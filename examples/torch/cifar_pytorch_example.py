import argparse
import os
from filelock import FileLock

import ray
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from resnet import ResNet18
from distml.util import override
from distml.strategy.allreduce_strategy import AllReduceStrategy
from distml.operator import TorchTrainingOperator


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


class CifarTrainingOperator(TorchTrainingOperator):
    @override(TorchTrainingOperator)
    def setup(self, config):
        # Create model.
        model = ResNet18(config)

        # Create optimizer.
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.1),
            momentum=config.get("momentum", 0.9))

        # Load in training and validation data.
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])  # meanstd transformation

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        with FileLock(".ray.lock"):
            train_dataset = CIFAR10(
                root="~/data",
                train=True,
                download=True,
                transform=transform_train)
            validation_dataset = CIFAR10(
                root="~/data",
                train=False,
                download=False,
                transform=transform_test)

        if config["test_mode"]:
            train_dataset = Subset(train_dataset, list(range(64)))
            validation_dataset = Subset(validation_dataset, list(range(64)))

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], num_workers=2)
        validation_loader = DataLoader(
            validation_dataset, batch_size=config["batch_size"], num_workers=2)

        # # Create scheduler.
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[150, 250, 350], gamma=0.1)

        # Create loss.
        criterion = nn.CrossEntropyLoss()
        print(criterion)
        # Register all components.
        # # self.model, self.optimizer, self.criterion, self.scheduler = \
        #     # self.register(models=model, optimizers=optimizer,
        #                   criterion=criterion, schedulers=scheduler)
        self.model, self.optimizer, self.criterion = \
            self.register(model=model, optimizer=optimizer,
                criterion=criterion)
        self.register_data(
            train_loader=train_loader, validation_loader=validation_loader)


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
        "--num-epochs", type=int, default=1, help="Number of epochs to train.")
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

    args, _ = parser.parse_known_args()
    num_cpus = 4 if args.smoke_test else None
    ray.init(address=args.address, num_cpus=num_cpus, log_to_driver=True)

    strategy = AllReduceStrategy(
        training_operator_cls=CifarTrainingOperator,
        initialization_hook=initialization_hook,
        world_size=args.num_workers,
        operator_config={
            "lr": 0.1,
            "test_mode": args.smoke_test,  # subset the data
            # this will be split across workers.
            "batch_size": 128 * args.num_workers
        })
    # pbar = trange(args.num_epochs, unit="epoch")
    # for i in pbar:
    #     info = {"num_steps": 1} if args.smoke_test else {}
    #     info["epoch_idx"] = i
    #     info["num_epochs"] = args.num_epochs
    #     # Increase `max_retries` to turn on fault tolerance.
    #     strategy.train(max_retries=1, info=info)
    #     # val_stats = trainer1.validate()
    #     # pbar.set_postfix(dict(acc=val_stats["val_accuracy"]))

    for i in range(args.num_epochs):
        strategy.train()
    print(strategy.validate())
    strategy.shutdown()
    print("success!")
