import re
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def decode_image(image_data, dim):
    image = tf.image.decode_jpeg(image_data, channels=3)
    # explicit size needed for TPU
    image = tf.reshape(image, [dim, dim, 3])

    return image


def augmentation_pipeline(image, label, seed):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_hue(image, 0.1, seed=seed)
    image = tf.image.random_saturation(image, 0, 1, seed=seed)
    return image, label


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


def get_training_dataset(TRAINING_FILENAMES, batch_size, augment=True):
    dataset = load_dataset(TRAINING_FILENAMES)
    dataset = dataset.cache()
    if augment:
        dataset = dataset.map(augmentation_pipeline,
                              num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_validation_dataset(VALIDATION_FILENAMES, batch_size, ):
    dataset = load_dataset(VALIDATION_FILENAMES)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    # prefetch next batch while validation (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_test_dataset(TEST_FILENAMES, batch_size, ):
    dataset = load_test_dataset(TEST_FILENAMES)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    # prefetch next batch while test (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
