import logging
from distml.operator.base_operator import TrainingOperator
import cupy as cp

try:
    import tensorflow as tf
    from tf.keras.losses import Loss
except ImportError:
    raise ImportError("Please install tensorflow following: "
                      "https://www.tensorflow.org/install.")

logger = logging.getLogger(__name__)

class TFTrainingOperator(TrainingOperator):
    """Class to define the training logic of a TensorFlow Model.

    Args:
        operator_config (dict): operator config specified by users.
    """

    def __init__(self, *, operator_config=None, **kwargs):
        super().__init__(operator_config=operator_config)
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
        self._use_gpu = bool(tf.config.list_physical_devices('GPU'))
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
        if not isinstance(model, tf.keras.Model):
            raise RuntimeError("`model` must be tf.keras.Models. "
                               "Got: {}".format(model))
        self._model = model
        # Shouldn't need this because tf runs on gpu by default
        '''
        if self._use_gpu:
            self._model.cuda()
        '''
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise RuntimeError("`optimizer` must be tf.keras.optimizers.Optimizer. "
                               "Got: {}".format(optimizer))

        # Note(Hao): this is problematic -- model and criterion are moved
        # to gpu but optimizer is constructed before the movement.
        # See: https://github.com/ray-project/ray/issues/15258
        self._optimizer = optimizer
        if criterion:
            if not isinstance(criterion, Loss):
                raise RuntimeError(
                    "`criterion` must be tf.keras.losses.Loss "
                    "Got: {}".format(self._criterion))
            self._criterion = criterion
            '''
            if self._use_gpu:
                self._criterion.cuda()
            '''
        # TODO(Hao): support lr schedulers
        return self._model, self._optimizer, self._criterion

    def register_data(self, *, train_loader=None, validation_loader=None):
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        # TODO(Hao): convert each data loader to be distributed

    def setup(self, *args, **kwargs):
        """Function that needs to be override by users."""
        raise NotImplementedError("Please override this function to register "
                                  "your model, optimizer, and criterion.")

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
        #model.train()

        '''
        if self._use_gpu:
            features = [
                feature.cuda(non_blocking=True) for feature in features
            ]
            target = target.cuda(non_blocking=True)
        '''

        # TODO(Hao): scope the code below using a timer?
        with tf.GradientTape() as tape:
            output = model(*features, training=True)
            loss = criterion(target, output)
        grads = tape.gradient(loss, model.trainable_weights)
        return loss.numpy(), grads

    def apply_updates(self, updates):
        """Set and apply the updates using the optimizer.step() in Torch.

        Args:
            updates (list): a list of parameter updates.
        """
        self.optimizer.apply_gradients(zip(updates, self.model.trainable_weights))

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
        '''
        if self._use_gpu:
            features = [
                feature.cuda(non_blocking=True) for feature in features
            ]
            target = target.cuda(non_blocking=True)
        '''

        output = model(*features, training=False)
        loss = criterion(target, output)

        # Todo(Hao): report accuracy instead loss here.
        batch_metric = {"val_loss": loss.numpy()}
        return batch_metric

    def get_states(self):
        raise NotImplementedError(
            "get_states is not support in tensorflow operator.")

    def load_states(self, checkpoint):
        """Load the states into the operator."""
        checkpointer = tf.train.Checkpoint(
            optimizer=self._optimizer,
            model=self._model,
            lr_scheduler = self._lr_scheduler,
        )

        self.model, self.optimizer, self._lr_scheduler = checkpointer.restore(checkpoint)

    def save_states(self, checkpoint_path):
        """Save the states to a file path."""
        checkpointer = tf.train.Checkpoint(
            optimizer=self._optimizer,
            model=self._model,
            lr_scheduler = self._lr_scheduler,
        )
        checkpointer.save(checkpoint_path)


    @staticmethod
    def to_cupy(tf_tensor):
        """Convert a tf GPU tensor to cupy tensor.

        Since now ray.util.collective natively support torch.Tensor,
        so we do nothing in this function.
        """
        if not isinstance(tf_tensor, tf.Tensor):
            raise RuntimeError("Expected tf.Tensor, but got: {}. "
                               .format(tf_tensor))
            dfcapsule = tf.experimental.dlpack.to_dlpack(tf_tensor)
            return cp.fromDlpack(dfcapsule)
