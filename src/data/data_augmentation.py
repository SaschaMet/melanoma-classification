# TF Augmentation functions. Source: https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
import os
import random
import tensorflow as tf

SEED = 1
DIM_RESIZE = int(os.environ["DIM_RESIZE"])


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


def rotate(x):
    """Rotation augmentation
    Args:
        x: Image

    Returns:
        Augmented image
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(x):
    crop_size = random.randrange(int(DIM_RESIZE*0.8), DIM_RESIZE)
    x = tf.image.random_crop(x, size=[crop_size, crop_size, 3])
    x = tf.image.resize(x, [DIM_RESIZE, DIM_RESIZE])
    return x


def augmentation_pipeline(image, label):
    augmentations = [flip, rotate, zoom]
    for f in augmentations:
        if random.randint(0, 10) >= 5:
            image = f(image)

    return image, label
