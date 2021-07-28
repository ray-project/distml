import argparse
import os
from filelock import FileLock

import ray

import ml_datasets

from thinc.api import fix_random_seed
from thinc.api import  Model, chain, strings2arrays, with_array,\
        HashEmbed, expand_window, Relu, Softmax, Adam, warmup_linear
from thinc.api import L2Distance

from distml.util import override
from distml.strategy.allreduce_strategy import AllReduceStrategy
from distml.operator.thinc_operator import ThincTrainingOperator


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"


"""
    Adapted based on: https://github.com/explosion/thinc/blob/master/examples/03_pos_tagger_basic_cnn.ipynb
"""
class POSTrainingOperator(ThincTrainingOperator):
    @override(ThincTrainingOperator)
    def setup(self, config):
        # Create model.
        width = 32
        vector_width = 16
        nr_classes = 17
        
        # fix random seed
        fix_random_seed(0)

        with Model.define_operators({">>": chain}):
            model = strings2arrays() >> with_array(
                HashEmbed(nO=width, nV=vector_width, column=0)
                >> expand_window(window_size=1)
                >> Relu(nO=width, nI=width * 3)
                >> Relu(nO=width, nI=width)
                >> Softmax(nO=nr_classes, nI=width)
            )
        # Create optimizer.
        optimizer = Adam(config["learning_rate"])

        # Load in training and validation data.
        with FileLock(".ray.lock"):
            (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ancora_pos_tags()
        train_loader = model.ops.multibatch(config["batch_size"], train_X, train_y, shuffle=True)
        dev_loader = model.ops.multibatch(config["batch_size"], dev_X, dev_y)

        # Initialize model and shape inference
        model.initialize(X=train_X[:5], Y=train_y[:5])

        # loss calculator
        loss_calc = L2Distance(normalize=False)
        # register model and data
        self.register(model=model, optimizer=optimizer, loss=loss_calc, use_gpu=config["use_gpu"])
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
        default=True,
        help="Enables GPU training")

    (train_X, train_y), (dev_X, dev_y) = ml_datasets.ud_ancora_pos_tags()
    #assert False
    args, _ = parser.parse_known_args()
    ray.init(address=args.address, log_to_driver=True)

    strategy = AllReduceStrategy(
        training_operator_cls=POSTrainingOperator,
        initialization_hook=initialization_hook,
        world_size=args.num_workers,
        operator_config={
            "learning_rate": 0.1,
            "use_gpu": args.use_gpu,
            # this will be split across workers.
            "batch_size": 128 * args.num_workers
        })

    for i in range(args.num_epochs):
        strategy.train()
    print(strategy.validate())
    strategy.shutdown()
    print("success!")
      
