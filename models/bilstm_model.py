"""This module contains DNN with different count of BiLSTM layers."""

from keras import layers, models

from utils.constants import IMAGE_WIDTH, IMAGE_HEIGHT, NEW_SHAPE
from utils.processing import characters_to_numbers

from .ctc_layer import CTCLayer


def initialize_bilstm_1():
    """
    The OCR model with 1 layer BiLSTM.

    Returns:
        A Keras model object representing the OCR model.
    """
    input_img = layers.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Reshape(target_shape=NEW_SHAPE, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # 1 BiLSTM layer
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(
        len(characters_to_numbers.get_vocabulary()) + 1,
        activation="softmax",
        name="dense2",
    )(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    return models.Model(
        inputs=[input_img, labels], outputs=output, name="OCR_with_BiLSTM_1"
    )


def initialize_bilstm_2():
    """
    The OCR model with 2 layers BiLSTM.

    Returns:
        A Keras model object representing the OCR model.
    """
    input_img = layers.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Reshape(target_shape=NEW_SHAPE, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # 2 BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(
        len(characters_to_numbers.get_vocabulary()) + 1,
        activation="softmax",
        name="dense2",
    )(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    return models.Model(
        inputs=[input_img, labels], outputs=output, name="OCR_with_BiLSTM_2"
    )
