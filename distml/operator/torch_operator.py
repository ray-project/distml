import logging

from distml.operator.base_operator import TrainingOperator

try:
    import torch
    from torch.nn.modules.loss import _Loss
except ImportError:
    raise ImportError("Please install PyTorch following: "
                      "https://pytorch.org/get-started/locally/.")

logger = logging.getLogger(__name__)


class TorchTrainingOperator(TrainingOperator):
    """Class to define the training logic of a PyTorch Model.

    Args:
        operator_config (dict): operator config specified by users.
    """

    def __init__(self, *, operator_config=None, **kwargs):
        super(TorchTrainingOperator,
              self).__init__(operator_config=operator_config)
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
        #   and support CPU training (with GLOO backend).
        self._use_gpu = torch.cuda.is_available()
        if not self._use_gpu:
            raise RuntimeError(
                "ray.util.distml now only supports GPU training.")
        self.setup(operator_config)

    def register(self,
                 *,
                 model,
                 optimizer,
                 criterion,
                 lr_scheduler=None,
                 **kwargs):
        # TODO(Hao): support custom training loop by allowing multiple model,
        # optimizer, e.g. for GAN training
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("`model` must be torch.nn.Modules. "
                               "Got: {}".format(model))
        self._model = model
        if self._use_gpu:
            self._model.cuda()
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError("`optimizer` must be torch.optim.Optimizer. "
                               "Got: {}".format(optimizer))

        # Note(Hao): this is problematic -- model and criterion are moved
        # to gpu but optimizer is constructed before the movement.
        # See: https://github.com/ray-project/ray/issues/15258
        self._optimizer = optimizer
        if criterion:
            if not isinstance(criterion, _Loss):
                raise RuntimeError(
                    "`criterion` must be torch.nn.module._Loss. "
                    "Got: {}".format(self._criterion))
            self._criterion = criterion
            if self._use_gpu:
                self._criterion.cuda()
        # TODO(Hao): support lr schedulers
        return self._model, self._optimizer, self._criterion

    def register_data(self, *, train_loader=None, validation_loader=None):
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        # TODO(Hao): convert each data loader to be distributed

    def derive_updates(self, batch):
        """Compute the parameter updates on a given batch of data.

        The `derive_updates` function should be called in conjunction with
        the next `apply_updates` function in order to finish one iteration
        of training.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        # TODO(Hao): 1. Add metric meters
        #             2. add lr_scheduler later.
        if not self.model:
            raise RuntimeError("Please set self.model at setup or override "
                               "this function for deriving gradient updates.")
        model = self.model
        if not self.optimizer:
            raise RuntimeError(
                "Please set self.optimizer at setup or override "
                "this function for deriving gradient updates.")
        optimizer = self.optimizer
        if not self.criterion:
            raise RuntimeError(
                "Please set self.criterion at setup or override "
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
        """Set and apply the updates using the optimizer.step() in Torch.

        Args:
            updates (dict): a dictionary of parameter name and updates.
        """
        self._set_gradients(self.model, updates)
        self.optimizer.step()

    def validate_batch(self, batch):
        """Perform validation over a data batch.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        if not self.model:
            raise RuntimeError("Please set self.model at setup or override "
                               "this function for validation.")
        model = self.model
        if not self.criterion:
            raise RuntimeError(
                "Please set self.criterion at setup or override "
                "this function for validation.")
        criterion = self.criterion
        *features, target = batch
        model.eval()
        if self._use_gpu:
            features = [
                feature.cuda(non_blocking=True) for feature in features
            ]
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(*features)
            loss = criterion(output, target)

        # Todo(Hao): report accuracy instead loss here.
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
            states.update({"lr_scheduler": self._lr_scheduler.state_dict()})
        return states

    def load_states(self, states=None, checkpoint=None):
        """Load the states into the operator."""
        if not states and not checkpoint:
            raise RuntimeError(
                "One of `states` and `checkpoint` should be provided. "
                "Got states: {}, checkpoint: {}.".format(states, checkpoint))
        if not states and checkpoint:
            states = self._load_from_checkpoint(checkpoint)
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        if self._lr_scheduler:
            self._lr_scheduler.load_state_dict(states["lr_scheduler"])
        self.load_custom_states(states["custom"])

    def _load_from_checkpoint(self, checkpoint):
        return torch.load(checkpoint)

    def save_states(self, checkpoint):
        """Save the states to a file path."""
        states = self.get_states()
        # TODO(Hao): test this.
        torch.save(states, checkpoint)

    @staticmethod
    def _get_gradients(model):
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

    @staticmethod
    def to_cupy(torch_tensor):
        """Convert a torch GPU tensor to cupy tensor.

        Since now ray.util.collective natively support torch.Tensor,
        so we do nothing in this function.
        """
        if not isinstance(torch_tensor, torch.Tensor):
            raise RuntimeError("Expected torch.Tensor, but got: {}. "
                               .format(torch_tensor))
        return torch_tensor

    @staticmethod
    def _set_gradients(model, grads):
        """Set the model gradients as grads."""
        for name, p in model.named_parameters():
            p.grad = grads[name]
            # if grads[name] is not None:
            #     if p.grad is not None:
            #         p.grad = torch.from_numpy(gradients[name]).
            #         to(p.grad.device)
            #     else:
            #         p.grad = torch.from_numpy(gradients[name])
