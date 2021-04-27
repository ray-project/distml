"""Abstract class for framework-specific training operators."""
from abc import ABCMeta
from abc import abstractmethod


class TrainingOperator(metaclass=ABCMeta):
    """Abstract class to define the training loop of a model.

    This class should be subclassed by the framework-specific operator implementations.
    For training, this class exposes two interfaces:
        - `derive_updates()`
        - `apply_updates()`
    in order for Ray collective backend to take over.

    For validation, this class exposes a single `validate_batch()` interface.
    The specific training and validation logic related with frameworks (JAX, PyTorch) is
    implemented in its subclasses

    Args:
        operator_config (dict): operator config specified by users.
    """
    def __init__(self, *args, operator_config=None, **kwargs):
        self._operator_config = operator_config

    @abstractmethod
    def register(self,
                 *,
                 model,
                 optimizer,
                 criterion,
                 **kwargs):
        """Register the model, optimizer, and criterion with the training operator.

        The function is instantiated in the framework-specific subclass. It
        is expected to be called by the user in self.setup().
        """
        raise NotImplementedError()

    @abstractmethod
    def register_data(self, *, train_loader=None, validation_loader=None):
        """Register batch-based data loaders."""
        raise NotImplementedError()

    def setup(self, operator_config):
        """Method to be override by users.

        In this method, the user should register the model, optimizer, criterion,
        and data loaders to the operator class via the `register()` method.
        """
        raise NotImplementedError()

    @abstractmethod
    def derive_updates(self, *args, **kwargs):
        """The sub-step that derives the gradient updates.

        This method should be instantiated by subclass operators.

        Returns:
            Tuple(loss, grads): A tuple that contains the loss value and
                the gradient updates.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply_updates(self, updates):
        """The sub-step that applies the updates.

        This method should be instantiated by subclass operators.

        Returns:
            None.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate_batch(self, *args, **kwargs):
        """Perform validation over a batch of validation data."""
        raise NotImplementedError()

    def get_custom_states(self, *args, **kwargs):
        """Functions to be optionally override by users to represent any custom states.

        See ``save_parameters`` for more details.
        """
        pass

    def load_custom_states(self, states, *args, **kwargs):
        """Functions to be optionally override by users to load any custom states.

        See ``load_parameters`` for more details.
        """
        pass

    @abstractmethod
    def save_states(self, checkpoint):
        """Save the states to a file path.

         This function shall be instantiated in framework-specific operator
         implementations.
         """
        raise NotImplementedError()

    @abstractmethod
    def get_states(self):
        """Return the states for the operator as a dict."""
        raise NotImplementedError()

    @abstractmethod
    def load_states(self, checkpoint):
        """Load the states from a file path.

        This functions shall be instantiated in framework-specific operators
        implementations.
        """
        raise NotImplementedError()

    def _get_train_loader(self):
        if not self._train_loader:
            raise RuntimeError("The operator does not have any registered train loader. "
                               "Please register the train loader via "
                               "`self.register_data()` inside the `self.setup()` function.")
        return self._train_loader

    def _get_validation_loader(self):
        if not self._validation_loader:
            raise RuntimeError("The operator does not have any registered validation loader. "
                               "Please register the validation loader via "
                               "`self.register_data()` inside the `self.setup()` function.")
        return self._validation_loader

    def _get_optimizer(self):
        if not self._optimizer:
            raise RuntimeError("The operator does not have any registered optimizer. "
                               "Please register the optimizer via "
                               "`self.register()` inside the `self.setup()` function.")
        return self._optimizer

    def _get_criterion(self):
        if not self._optimizer:
            raise RuntimeError("The operator does not have any registered criterion. "
                               "Please register the criterion via "
                               "`self.register()` inside the `self.setup()` function.")
        return self._criterion
