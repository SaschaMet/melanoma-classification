import re
import os
import numpy as np
import tensorflow as tf

from data.data_augmentation import augmentation_pipeline

SEED = 1
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = int(os.environ["BATCH_SIZE"])


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def decode_image(image_data):
    dim = int(os.environ["DIM"])
    image = tf.image.decode_jpeg(image_data, channels=3)
    # explicit size needed for TPU
    image = tf.reshape(image, [dim, dim, 3])

    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        # tf.string means bytestring
        "image": tf.io.FixedLenFeature([], tf.string),
        # shape [] means single element
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['target']
    return image, label  # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = example['image_name']
    return image, label


def load_dataset(filenames):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    opt = tf.data.Options()
    opt.experimental_deterministic = False

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(opt)

    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def load_test_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(read_unlabeled_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset


def get_training_dataset(filenames, batch_size=BATCH_SIZE, augment=True):
    dataset = load_dataset(filenames)
    dataset = dataset.cache()
    dataset = dataset.shuffle(BATCH_SIZE * 3, seed=SEED,
                              reshuffle_each_iteration=True)
    if augment:
        dataset = dataset.map(augmentation_pipeline,
                              num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.batch(batch_size)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_validation_dataset(filenames, batch_size=BATCH_SIZE):
    dataset = load_dataset(filenames)
    dataset = dataset.cache()
    dataset = dataset.shuffle(BATCH_SIZE * 3, seed=SEED,
                              reshuffle_each_iteration=False)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_prediction_validation_dataset(filenames, batch_size=BATCH_SIZE):
    dataset = load_dataset(filenames)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_test_dataset(filenames, batch_size=BATCH_SIZE):
    dataset = load_test_dataset(filenames)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
