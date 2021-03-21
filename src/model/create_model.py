import tensorflow as tf
import efficientnet.keras as efn


def create_model(num_classes, image_dimensions, initial_bias, fine_tune=False):
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
