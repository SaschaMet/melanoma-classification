# TF Augmentation functions. Source: https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
import os
import random
import tensorflow as tf

SEED = 1
DIM = int(os.environ["DIM"])


def flip(x):
    """Flip augmentation
    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    return x


def color(x):
    """Color augmentation
    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08, seed=SEED)
    x = tf.image.random_saturation(x, 0.6, 1.6, seed=SEED)
    x = tf.image.random_brightness(x, 0.05, seed=SEED)
    x = tf.image.random_contrast(x, 0.7, 1.3, seed=SEED)
    return x


def rotate(x):
    """Rotation augmentation
    Args:
        x: Image

    Returns:
        Augmented image
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(x):
    crop_size = random.randrange(int(DIM*0.75), DIM)
    x = tf.image.random_crop(x, size=[crop_size, crop_size, 3])
    x = tf.image.resize(x, (DIM, DIM))
    x = tf.cast(x, tf.uint8)
    return x


def augmentation_pipeline(image, label):
    augmentations = [flip, color, rotate, zoom]
    for f in augmentations:
        if random.randint(1, 10) >= 5:
            image = f(image)

    return image, label
