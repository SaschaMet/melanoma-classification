import re
import os
import numpy as np
import tensorflow as tf
from functools import partial

from data.data_augmentation import augment_image

SEED = 1
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
DIM = int(os.environ["DIM"])
DIM_RESIZE = int(os.environ["DIM_RESIZE"])

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    1./255)
resizing_layer = tf.keras.layers.experimental.preprocessing.Resizing(
    DIM_RESIZE, DIM_RESIZE)


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.cache()  # cache ds for performance gains
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # normalize the image so the values are between 0 and 255
    dataset = dataset.map(lambda x, y: (
        normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    # resize the images to the same height and width
    dataset = dataset.map(lambda x, y: (
        resizing_layer(x), y), num_parallel_calls=AUTOTUNE)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_training_dataset(filenames, number_of_images, batch_size=BATCH_SIZE, augment=True):
    dataset = load_dataset(filenames, labeled=True)
    if augment:
        dataset = dataset.map(lambda x, y: (augment_image(
            x, augment=augment), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(
        number_of_images * 2, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_validation_dataset(filenames, batch_size=BATCH_SIZE, repeat=False, ordered=False):
    dataset = load_dataset(filenames, labeled=True, ordered=ordered)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_test_dataset(filenames, batch_size=BATCH_SIZE, ordered=True):
    dataset = load_dataset(filenames, labeled=False, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
