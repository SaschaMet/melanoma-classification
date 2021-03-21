import tensorflow as tf
import efficientnet.keras as efn


def create_model(num_classes, image_dimensions, initial_bias=False, fine_tune=False):
    """Creates a EfficientNetB5 model

    Args:
        num_classes (int): Number of classes to predict
        image_dimensions (int): Image dimensions
        initial_bias (bool, optional): Initial bias of the output layer
        fine_tune (bool, optional): True sets the whole model to trainable. Defaults to False.

    Returns:
        EfficientNetB5 sequential model
    """
    base_model = efn.EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_shape=(image_dimensions, image_dimensions, 3),
    )
    base_model.trainable = fine_tune

    if initial_bias is not None:
        output_bias = tf.keras.initializers.Constant(initial_bias)

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(
            num_classes, activation='sigmoid', bias_initializer=output_bias)
    ])

    return model
