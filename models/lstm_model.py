from keras import Sequential, layers, models
from utils.constants import IMAGE_WIDTH, IMAGE_HEIGHT, NEW_SHAPE
from utils.processing import characters_to_numbers
from .ctc_layer import CTCLayer


def initialize_LSTM_1():
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    model = Sequential([
        layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image", dtype="float32"),
        layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pooling1"),
        layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pooling2"),
        layers.Reshape(target_shape=NEW_SHAPE, name="reshape"),
        layers.Dense(64, activation="relu", name="dense1"),
        layers.Dropout(0.2),
        layers.LSTM(128, return_sequences=True, dropout=0.25),
        layers.Dense(len(characters_to_numbers.get_vocabulary()) + 1, activation="softmax", name="dense2")
    ])
    output = CTCLayer(name="ctc_loss")(labels, model.output)
    return models.Model(inputs=[model.input, labels], outputs=output, name="OCR_with_LSTM_1")

def initialize_LSTM_2():
  labels = layers.Input(name="label", shape=(None,), dtype="float32")
  model = Sequential([
      layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image", dtype="float32"),
      layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv1"),
      layers.MaxPooling2D((2, 2), name="pooling1"),
      layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2"),
      layers.MaxPooling2D((2, 2), name="pooling2"),
      layers.Reshape(target_shape=NEW_SHAPE, name="reshape"),
      layers.Dense(64, activation="relu", name="dense1"),
      layers.Dropout(0.2),
      layers.LSTM(128, return_sequences=True, dropout=0.25),
      layers.LSTM(64, return_sequences=True, dropout=0.25),
      layers.Dense(len(characters_to_numbers.get_vocabulary()) + 1, activation="softmax", name="dense2")
  ])
  output = CTCLayer(name="ctc_loss")(labels, model.output)
  return models.Model(inputs=[model.input, labels], outputs=output, name="OCR_with_LSTM_2")
