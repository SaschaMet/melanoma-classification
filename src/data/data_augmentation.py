import tensorflow as tf


def augmentation_pipeline(image, label, seed=1):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_hue(image, 0.1, seed=seed)
    image = tf.image.random_saturation(image, 0, 1, seed=seed)
    return image, label
