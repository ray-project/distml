import os
import logging

from ray.util.distml.base_operator import TrainingOperator

import torch
from torch.nn.modules.loss import _Loss

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class TorchTrainingOperator(TrainingOperator):
    """Class to define the training logic of a PyTorch Model."""
    def __init__(self,
                 *,
                 operator_config=None,
                 **kwargs):

        # Initialize the PyTorch training operator
        super(TorchTrainingOperator, self).__init__(operator_config=operator_config)
        # Should be set by users in the `register` function.
        self.model = None
        self.optimizer = None
        self.criterion = None
        # Models, optimizers, and criterion registered by users.
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._lr_scheduler = None

        # Data loaders for training and validation, registered by users.
        self._train_loader = None
        self._validation_loader = None

        # TODO(Hao): lift the use_gpu attributes below to operator arguments,
        #            and support CPU training (with GLOO backend).
        self._use_gpu = torch.cuda.is_available()
        if not self._use_gpu:
            raise RuntimeError("ray.util.distml now only supports GPU training.")
        self.setup(operator_config)


    def register(self,
                 *,
                 model,
                 optimizer,
                 criterion=None,
                 lr_scheduler=None,
                 **kwargs):
        # TODO(Hao): support custom training loop by allowing multiple model, optimizer,
        # e.g. GAN case.
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("`model` must be torch.nn.Modules. "
                               "Got: {}".format(model))
        self._model = model
        if self._use_gpu:
            self._model.cuda()
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError("`optimizer` must be torch.optim.Optimizer. "
                               "Got: {}".format(optimizer))
        # Note(Hao): this is problematic -- model and criterion are moved to gpu
        # but optimizer is constructed before the movement.
        # See: https://github.com/ray-project/ray/issues/15258
        self._optimizer = optimizer
        if criterion:
            if not isinstance(criterion, _Loss):
                raise RuntimeError("`criterion` must be torch.nn.module._Loss. "
                                   "Got: {}".format(self._criterion))
            self._criterion = criterion
            if self._use_gpu:
                self._criterion.cuda()
        if lr_scheduler:
            # TODO(Hao): register schedulers
            self._lr_schedulers = lr_scheduler
        return self._model, self._optimizer, self._criterion

    def register_data(self, *, train_loader=None, validation_loader=None):
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        # TODO(Hao): convert each data loader to be distributed

    def derive_updates(self, batch):
        """Compute the parameter updates on a given batch of data.

        The `derive_updates` function should be called in conjunction with
        the next `apply_updates` function in order to finish one iterator
        of training.
        """
        # TODO(Hao): 1. Add metric meters
        #             2. add lr_scheduler later.
        if not self.model:
            raise RuntimeError("Please set self.model at setup or override "
                               "this function for deriving gradient updates.")
        model = self.model
        if not self.optimizer:
            raise RuntimeError("Please set self.optimizer at setup or override "
                               "this function for deriving gradient updates.")
        optimizer = self.optimizer
        if not self.criterion:
            raise RuntimeError("Please set self.criterion at setup or override "
                               "this function for deriving gradient updates.")
        criterion = self.criterion
        *features, target = batch
        model.train()

        if self._use_gpu:
            features = [
                feature.cuda(non_blocking=True) for feature in features
            ]
            target = target.cuda(non_blocking=True)

        # TODO(Hao): scope the code below using a timer?
        output = model(*features)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        grads = self._get_gradients(model)
        return loss.item(), grads

    def apply_updates(self, updates):
        """Apply the updates using the optimizer.step() in Torch."""
        self._set_gradients(self.model, updates)
        self.optimizer.step()

    def validate(self, validation_iterator):
        """Perform validation over validation dataset, represented as an iterator."""
        metric = {}
        for i, batch in enumerate(validation_iterator):
            metric_per_batch = self.validate_step(batch)
            metric.update(metric_per_batch)
        return metric

    def validate_step(self, batch):
        """Perform validation over a data batch."""
        # TODO(Hao): implement this method, referring to RaySGD.
        if not self.model:
            raise RuntimeError("Please set self.model at setup or override "
                               "this function for validation.")
        model = self.model
        if not self.criterion:
            raise RuntimeError("Please set self.criterion at setup or override "
                               "this function for validation.")
        criterion = self.criterion
        *feature, target = batch
        output = model(*feature)
        loss = criterion(output, target)
        batch_metric = {"val_loss": loss.item()}
        return batch_metric

    def get_states(self):
        """Return the states of this training operator."""
        states = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "custom": self.get_custom_states()
        }
        if self._lr_scheduler:
            states.update({
                "lr_scheduler": self._lr_scheduler.state_dict()}
            )
        return states

    def load_states(self, states=None, checkpoint=None):
        """Load the states into the operator."""
        if not states and not checkpoint:
            raise RuntimeError("One of `states` and `checkpoint` should be provided. "
                               "Got states: {}, checkpoint: {}.".format(states, checkpoint))
        if not states and checkpoint :
            states = self._load_from_checkpoint(checkpoint)
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        if self._lr_scheduler:
            self._lr_scheduler.load_state_dict(states["lr_scheduler"])
        self.load_custom_states(states["custom"])

    def save_states(self, checkpoint):
        """Save the states to a file path."""
        states = self.get_states()
        # TODO(Hao): test this.
        torch.save(states, checkpoint)

    def _get_gradients(self, model):
        """Return the gradient updates of the model as a Python dict.

        Returns:
            grads (dict): a dictionary of parameter name and grad tensors.
        """
        grads = {}
        for name, p in model.named_parameters():
            # grad = None if p.grad is None else p.grad.data
            # grad = None if p.grad is None else p.grad
            grads[name] = p.grad.data
            logger.debug("grad name: {}, grad type: {}, grad value: {} "
                         .format(name, type(grads[name]), grads[name]))
        return grads

    def to_cupy(self, torch_tensor):
        """Convert a torch GPU tensor to cupy tensor.

        Since now ray.util.collective natively support torch.Tensor,
        so we do nothing in this function.
        """
        if not isinstance(torch_tensor, torch.Tensor):
            raise RuntimeError("Expected torch.Tensor, but got: {}. "
                               .format(torch_tensor))
        return torch_tensor

    def _set_gradients(self, model, grads):
        """Set the model gradients as grads."""
        for name, p in model.named_parameters():
            p.grad = grads[name]
            # if grads[name] is not None:
            #     if p.grad is not None:
            #         p.grad = torch.from_numpy(gradients[name]).to(p.grad.device)
            #     else:
            #         p.grad = torch.from_numpy(gradients[name])
