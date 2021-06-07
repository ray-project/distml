import os

from filelock import FileLock

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from distml.util import override
from distml.strategy.allreduce_strategy import AllReduceStrategy
from distml.strategy.ps_strategy import ParameterServerStrategy
from distml.operator.torch_operator import TorchTrainingOperator

__all__ = ["make_torch_ar_strategy", "make_torch_ps_strategy", "ToyOperator"]


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    os.environ["NCCL_SHM_DISABLE"] = "1"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


class ToyModel(nn.Module):
    def __init__(self, config):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class ToyOperator(TorchTrainingOperator):
    @override(TorchTrainingOperator)
    def setup(self, config):
        # Create model.
        model = ToyModel(config)

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

        # Create loss.
        criterion = nn.CrossEntropyLoss()

        self.model, self.optimizer, self.criterion = self.register(
            model=model, optimizer=optimizer, criterion=criterion)

        self.register_data(
            train_loader=train_loader, validation_loader=validation_loader)


def make_torch_ar_strategy(world_size=2):
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


def make_torch_ps_strategy(num_ps=2, num_worker=2):
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
