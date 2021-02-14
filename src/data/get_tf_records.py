import re
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE
cfg = dict(
    read_size=1024,
    crop_size=500,
    net_size=448,
)


def verify_tf_records(base_path, TFRECORDS):
    _, ax = plt.subplots(5, 2, figsize=(10, 25))
    ds = get_dataset(TFRECORDS, labeled=True).unbatch().take(5)
    for idx, item in enumerate(ds):
        ax[idx][0].imshow(item[0][0])
        original = plt.imread(os.path.join(
            base_path, 'data', 'train', item[1].numpy().decode("utf-8") + '.jpg'))
        ax[idx][1].imshow(original)
        print('Sex: %s, Age: %s, Site: %s' %
              (item[0][1], item[0][2], item[0][3]))


def read_labeled_tfrecord(example, return_image_name):
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
    return example['image'], example['sex'], example['age_approx'], example['anatom_site_general_challenge'], example['image_name'] if return_image_name else example['target']


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['sex'], example['age_approx'], example['anatom_site_general_challenge'], example['image_name'] if return_image_name else 0


def prepare_image(img, augment=True):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])
    img = tf.cast(img, tf.float32) / 255.0
    # img = tf.image.central_crop(img, cfg['crop_size'] / cfg['read_size'])
    # img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])
    # img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])
    return img


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
         for filename in filenames]
    return np.sum(n)


def get_dataset(files, labeled=True, return_image_names=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO).cache()

    if labeled:
        ds = ds.map(lambda example: read_labeled_tfrecord(
            example, return_image_names), num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(
            example, return_image_names), num_parallel_calls=AUTO)

    ds = ds.map(lambda img, sex, age, site, label: tuple([tuple([prepare_image(img), sex, age, site]), label]),
                num_parallel_calls=AUTO)
    ds = ds.batch(32)
    ds = ds.prefetch(AUTO)
    return ds
