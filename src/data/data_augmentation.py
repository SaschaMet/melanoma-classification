import os
import math
import random
import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 1
DIM_RESIZE = int(os.environ["DIM_RESIZE"])

AUGMENTATION_CONFIG = dict(
    rot=180.0,
    shr=1.5,
    hzoom=6.0,
    wzoom=6.0,
    hshift=6.0,
    wshift=6.0
)


def color(x):
    """Color augmentation
    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_saturation(x, 0.7, 1.7, seed=SEED)
    x = tf.image.random_brightness(x, 0.3, seed=SEED)
    x = tf.image.random_contrast(x, 0.5, 1.5, seed=SEED)
    return x


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


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """returns 3x3 transform matrix which transforms indicies
        Source: https://github.com/Masdevallia/3rd-place-kaggle-siim-isic-melanoma-classification/blob/master/kaggle_notebooks/melanoma-classification-model-training.ipynb
    """

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1,   s1,   zero,
                                   -s1,  c1,   zero,
                                   zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one,  s2,   zero,
                                zero, c2,   zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero,
                               zero,            one/width_zoom, zero,
                               zero,            zero,           one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift,
                                zero, one,  width_shift,
                                zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, cfg=AUGMENTATION_CONFIG):
    """ Returns a randomly rotated, sheared, zoomed, and shifted image

    Args:
        image: one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        cfg (dict, optional): Augmentation config. Defaults to AUGMENTATION_CONFIG.

    Returns:
        image: image randomly rotated, sheared, zoomed, and shifted
    """
    DIM = DIM_RESIZE
    XDIM = DIM % 2  # fix for size 331

    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32')
    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM//2, -DIM//2, -1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0, ], DIM//2-1+idx2[1, ]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM_RESIZE, DIM_RESIZE, 3])


def augment_image(image, augment=True):
    """Randomly applies a image transformation to an input image

    Args:
        image (image)
        augment (bool, optional): Apply augmentation or not. Defaults to True.

    Returns:
        image
    """
    augmentations = [transform, color, rotate, flip]
    if augment:
        # Data augmentation
        for f in augmentations:
            if random.randint(1, 10) >= 5:
                image = f(image)
    return image
