import logging

from distml.operator.base_operator import TrainingOperator

try:
    import thinc
except ImportError:
    raise ImportError("Please install Thinc following: "
                      "https://thinc.ai/docs/install")

logger = logging.getLogger(__name__)

class ThincTrainingOperator(TrainingOperator):


    def __init__(self, *, operator_config=None, **kwargs):
        super(ThincTrainingOperator, 
            self).__init__(operator_config=operator_config)
        self.model = None
        self.optimizer = None
        self.loss_calculator = None

        self._model = None
        self._optimizer = None
        self._loss_calculator = None

        self._use_gpu = None
        self.setup(operator_config)
    
    def setup(self, *args, **kwargs):
        """Function that needs to be override by users."""
        raise NotImplementedError("Please override this function to register "
                                  "your model, optimizer, and criterion.")


    def register(self, *, model, optimmizer, loss=None, use_gpu=True, **kwargs):
        if not isinstance(model, thinc.api.Model):
            raise RuntimeError("`model` must be thinc.api.Models. "
                               "Got: {}".format(model))
        self._model = model
        if not use_gpu:
            raise RuntimeError(
                "ray.util.distml now only supports GPU training.")
        try:
            thinc.api.require_gpu()
        except:
            raise RuntimeError("ray.util.distml now only supports GPU training.")
        
        if not isinstance(optimmizer, thinc.api.Optimizer):
            raise RuntimeError("`optimizer` must be thinc.api.Optimmizer. "
                               "Got: {}".format(optimizer))
        self._optimizer = optimmizer
        if loss:
            if not isinstance(loss, thinc.api.Loss):
                raise RuntimeError("`loss` must be thinc.api.Loss. "
                                   "Got: {}".format(loss))
            self._loss_calculator = loss
        return self._model, self._optimizer, self._loss_calculator
    
    def register_data(self, *, train_loader=None, validation_loader=None):
        self._train_loader = train_loader
        self._validation_loader = validation_loader

    def derive_updates(self, batch):
        """Compute the parameter updates on a given batch of data.

        The `derive_updates` function should be called in conjunction with
        the next `apply_updates` function in order to finish one iteration
        of training.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        if not self.model:
            raise RuntimeError("Please set self.model at setup or override "
                               "this function for deriving gradient updates.")
        model = self.model
        if not self.optimizer:
            raise RuntimeError(
                "Please set self.optimizer at setup or override "
                "this function for deriving gradient updates.")
        optimizer = self.optimizer
        if not self.loss_calculator:
            raise RuntimeError(
                "Please set self.loss_calculator at setup or override "
                "this function for deriving gradient updates.")
        loss_calculator = self._loss_calculator
        *features, target = batch
        
        output, backprop = model.begin_update(*features)
        grad, loss = loss_calculator(output, target)
        backprop(grad)
        grads = self._get_gradients(model)
        return loss.item(), grads
    

    def apply_updates(self, updates):
        self._set_gradients(self.model, updates)
        # TODO(Fu): note sure if this is correct
        model.finish_update(optimizer) 

    
    def validate_batch(self, batch):
        """Perform validation over a data batch.

        Args:
            batch (tuple): a data batch containing a feature/target pair.
        """
        if not self.model:
            raise RuntimeError("Please set self.model at setup or override "
                               "this function for validation.")
        model = self.model
        if not self.loss_calculator:
            raise RuntimeError(
                "Please set self.loss_calculator at setup or override "
                "this function for validation.")
        loss_calculator = self.loss_calculator
        *features, target = batch

        output = model.predict(*features)
        loss = loss_calculator(output, target)
        batch_metric = {"val_loss": loss.item()}
        return batch_metric
    
    @staticmethod
    def _get_gradients(model):
        """Return the gradient updates of the model as a Python dict.

        Returns:
            grads (dict): a dictionary of parameter name and grad tensors.
        """
        grads = {}
        for name, p in model.param_names:
            grads[name] = model.get_grad(name)
            logger.debug("grad name: {}, grad type: {}, grad value: {} "
                         .format(name, type(grads[name]), grads[name]))
        return grads
    
    @staticmethod
    def _set_gradients(model, grads):
        """Set the model gradients as grads."""
        for name, p in model.param_names:
            model.set_grad(name, grads[name])