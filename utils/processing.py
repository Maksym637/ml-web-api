import string

import numpy as np

import tensorflow as tf
from keras import layers, backend, models
from .constants import MAX_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE

CHARACTERS = sorted(list(string.ascii_lowercase) + list(string.digits))

characters_to_numbers = layers.StringLookup(
    vocabulary=CHARACTERS, mask_token=None
)

numbers_to_characters = layers.StringLookup(
    vocabulary=characters_to_numbers.get_vocabulary(), mask_token=None, invert=True
)

def define_model(model):
    return models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LENGTH]
    output_text = []

    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(numbers_to_characters(res)).numpy().decode("utf-8")
        output_text.append(res)
      
    return output_text

def encode_single_sample_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}

def prepare_data(file_path):
    image = [file_path]
    data = tf.data.Dataset.from_tensor_slices(np.array(image))
    data = (data.map(encode_single_sample_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE))
    return data
