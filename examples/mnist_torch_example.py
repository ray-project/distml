import os
import argparse
import functools
from tqdm import trange

import torch 
import torch.optim as optim
import torch.nn.functional as f
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torchvision

from filelock import FileLock
from distml.strategy.util import ThroughputCollection

import numpy as np
import numpy.random as npr

from jax_util.datasets import mnist

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1  wiki_torch_example.py

class Dataloader:
    def __init__(self, data, target, batch_size=128, shuffle=False):
        '''
        data: shape(width, height, channel, num)
        target: shape(num, num_classes)
        '''
        self.data = data
        self.target = target
        self.batch_size = batch_size
        num_data = self.target.shape[0]
        num_complete_batches, leftover = divmod(num_data, batch_size)
        self.num_batches = num_complete_batches + bool(leftover)
        self.shuffle = shuffle

    def synth_batches(self):
        num_imgs = self.target.shape[0]
        rng = npr.RandomState(npr.randint(10))
        perm = rng.permutation(num_imgs) if self.shuffle else np.arange(num_imgs)
        for i in range(self.num_batches):
            batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
            img_batch = self.data[batch_idx, :, :, :]
            label_batch = self.target[batch_idx]
            yield img_batch, label_batch

    def __iter__(self):
        return self.synth_batches()

    def __len__(self):
        return self.num_batches


class Net(nn.Module):
    def __init__(self , model_name):
        super(Net, self).__init__()
        #取掉model的后两层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        if args.model_name == "resnet18":
            base_model = torchvision.models.resnet18(pretrained=False)
            self.resnet_layer = nn.Sequential(*list(base_model.children())[1:-2])
            inplane = 512
        elif args.model_name == "resnet101":
            base_model = torchvision.models.resnet101(pretrained=False)
            self.resnet_layer = nn.Sequential(*list(base_model.children())[1:-2])
            inplane = 2048
        else:
            raise RuntimeError("Unrecognized model name. Got {}.".format(args.model_name))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear_layer = nn.Linear(inplane, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


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
        default=4,
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
    parser.add_argument(
        "--trainer", type=str, default="ar", help="Trainer type, Optional: ar, ps")
    parser.add_argument(
        "--local_rank", type=int, default=0)
    parser.add_argument(
        "--distributed", action="store_false", default=False)
    parser.add_argument(
        "--model-name", type=str, default="resnet18")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

    args, _ = parser.parse_known_args()

    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("Using world size ", world_size)
    args.distributed = world_size > 1
    device = torch.device(args.local_rank%8)

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        job_name = f"mnist_torch_ar_{world_size}workers"
    else:
        job_name = f"mnist_torch_ar_{world_size}workers"

    torch.cuda.set_device(device)


    batch_size = 128

    with FileLock(".ray.lock"):
        train_images, train_labels, test_images, test_labels = mnist()
        
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)


    train_loader = Dataloader(train_images, train_labels, batch_size=batch_size, shuffle=True)
    test_loader = Dataloader(test_images, test_labels, batch_size=batch_size)
    
    model = Net(args.model_name)

    if args.distributed:
        model = DistributedDataParallel(model.to(device), device_ids=[device])
    else:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    collector = ThroughputCollection(batch_size=batch_size*world_size, job_name=job_name)

    for i in range(5):
        for idx, batch in enumerate(train_loader):
            with collector.record("train_batch"):
                inputs, targets = batch
                inputs = torch.Tensor(inputs).to(device)
                targets = torch.LongTensor(targets).to(device).argmax(dim=-1)
                logits = model(inputs)
                loss = criterion(logits, targets)

                model.zero_grad()
                loss.backward()

                optimizer.step()

            if not args.local_rank and idx % 10 == 0:
                print('Loss step {}: '.format(idx), loss.item())
