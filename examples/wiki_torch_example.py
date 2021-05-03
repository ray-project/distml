import os
import argparse
import functools

import torch 
import torch.optim as optim
import torch.nn.functional as f
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from distml.strategy.util import ThroughputCollection

from flax_util.datasets import make_wiki_train_loader, tf2numpy
from flax_util.models import Bert

from jax_util.datasets import _one_hot

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForPreTraining
# pip install git+https://github.com/Ezra-H/transformers.git

import torch.distributed as dist

# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1  wiki_torch_example.py


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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

    args, _ = parser.parse_known_args()

    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("Using world size ", world_size)
    args.distributed = world_size > 1

    print("using distributed", args.distributed )
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        job_name = f"wiki_torch_ar_{world_size}workers"
    else:
        job_name = f"benchmark_wiki_torch"

    device = torch.device(args.local_rank%8)
    torch.cuda.set_device(device)


    batch_size = 8
    train_loader = make_wiki_train_loader(batch_size=batch_size)

    # config = BertConfig.from_pretrained('bert-large-uncased')
    config = BertConfig()
    model = BertForPreTraining(config)
        
    if args.distributed:
        model = DistributedDataParallel(model.to(device), device_ids=[device])
    else:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss(reduction="none")

    collector = ThroughputCollection(batch_size=batch_size*world_size, job_name=job_name)

    def loss_func(logits, masked_positions, masked_lm_ids, masked_lm_weights, next_sentence_label):
        mask_logits = torch.cat([logit[masked_position] for logit, masked_position in zip(logits[0], masked_positions)])
        mask_logits = f.log_softmax(mask_logits, dim=-1)
        sentence_logits = f.log_softmax(logits[1], dim=-1)

        loss1 = criterion(sentence_logits, next_sentence_label)

        loss2 = criterion2(mask_logits, masked_lm_ids.view(-1))
        loss2 = torch.mean(loss2 * masked_lm_weights.view(-1))
        loss = (loss1 + loss2)/2
        return loss


    for i in range(1):
        for idx, batch in enumerate(train_loader):
            with collector.record("train_batch"):
                batch = tf2numpy(batch)
                input_ids = batch[0]["input_word_ids"]
                attention_mask = batch[0]["input_mask"]
                token_type_ids =  batch[0]["input_type_ids"]
                masked_positions = batch[0]["masked_lm_positions"]
                masked_lm_ids = batch[0]["masked_lm_ids"]
                masked_lm_weights = batch[0]["masked_lm_weights"]
                next_sentence_label = batch[0]["next_sentence_labels"]

                input_ids = torch.LongTensor(input_ids).to(device)
                attention_mask = torch.LongTensor(attention_mask).to(device)
                token_type_ids =  torch.LongTensor(token_type_ids).to(device)
                masked_positions = torch.LongTensor(masked_positions).to(device)
                masked_lm_ids = torch.LongTensor(masked_lm_ids).to(device)
                masked_lm_weights = torch.LongTensor(masked_lm_weights).to(device)
                next_sentence_label = torch.LongTensor(next_sentence_label).view(-1).to(device)
                position_ids = None

                # TODO(HUI): record.
                logits = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids)

                loss = loss_func(logits, masked_positions, masked_lm_ids, masked_lm_weights, next_sentence_label)

                model.zero_grad()
                loss.backward()

                optimizer.step()

            if not args.local_rank and idx % 10 == 0:
                print('Loss step {}: '.format(idx), loss.item())
