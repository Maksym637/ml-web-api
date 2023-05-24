"""Module that represents the CTC layer."""

import tensorflow as tf
from keras import layers, backend

class CTCLayer(layers.Layer):
    """
    Custom layer for computing the Connectionist Temporal Classification (CTC) loss.

    This layer calculates the CTC loss between the true labels and predicted outputs.

    Attributes:
        loss_function: The backend CTC loss function
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """
        Computes the CTC loss between the true labels and predicted outputs.

        Args:
            y_true (tf.Tensor): The true labels tensor.
            y_pred (tf.Tensor): The predicted outputs tensor.

        Returns:
            tf.Tensor: The predicted outputs tensor.
        """
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_function(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
