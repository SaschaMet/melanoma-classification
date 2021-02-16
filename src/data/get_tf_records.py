import re
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

strategy = tf.distribute.get_strategy()
REPLICAS = strategy.num_replicas_in_sync
AUTOTUNE = tf.data.experimental.AUTOTUNE


def validate_tf_records(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(6):
            finding = "Benign" if labels[i].numpy() == 0 else "Malignant"
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(finding)
            plt.axis("off")


def read_labeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['target']


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_name'] if return_image_name else 0


def prepare_image(img, augment=True, dim=256):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)

    img = tf.image.resize(img, [dim, dim])

    return img


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def get_dataset(files, augment=False, shuffle=False, repeat=False,
                labeled=True, return_image_names=True, batch_size=16, dim=256):

    ignore_order = tf.data.Options()
    if not labeled:
        ignore_order.experimental_deterministic = False  # disable order, increase speed

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(ignore_order)

    ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(
            example, return_image_names), num_parallel_calls=AUTOTUNE)

    ds = ds.map(lambda img, imgname_or_label: (
        prepare_image(img, augment=augment, dim=dim), imgname_or_label),
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(batch_size * REPLICAS)
    ds = ds.prefetch(AUTOTUNE)
    return ds
