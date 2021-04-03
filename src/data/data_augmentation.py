import math
import random
import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 1
ROT_ = 180.0
SHR_ = 2
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Args:
        rotation (int)
        shear (int)
        height_zoom (int)
        width_zoom (int)
        height_shift (int)
        width_shift (int)

    Returns:
        3x3 transformmatrix which transforms indicies
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


def transform(image, DIM):
    """Here we apply some manual augmentations that cannot be done with tf.image,
    such as shearing, zooming and translation. Rotation can be done in tf.image but only in factors of 90 degrees,
    so we do it manually instead.
    Source: https://www.kaggle.com/teyang/melanoma-detection-using-effnet-and-meta-data#5.-Train-and-Evaluate-Model

    Args:
        image (array): Input image is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        DIM (int): Image dimension

    Returns:
        Image: Image randomly rotated, sheared, zoomed, and shifted
    """
    XDIM = DIM % 2  # fix for size 331

    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32')
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32')

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

    return tf.reshape(d, [DIM, DIM, 3])


def color(x):
    """Color augmentation
    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.05)
    x = tf.image.random_brightness(x, 0.1, seed=SEED)
    x = tf.image.random_contrast(x, 0.6, 1.4, seed=SEED)
    x = tf.image.random_saturation(x, 0.6, 1.4, seed=SEED)
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


def augment_image(image, augment=True):
    """Apply data augmentation to image to

    Args:
        image (image array): Input image
        augment (bool, optional): Wether to apply augmentation or not. Defaults to True.

    Returns:
        Image
    """
    augmentations = [color, flip, transform]
    if augment:
        # Data augmentation
        for f in augmentations:
            if random.randint(1, 10) <= 5:
                image = f(image)
    return image
