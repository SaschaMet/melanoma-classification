import tensorflow as tf


def create_model(NUM_CLASSES, DIM):
    i = tf.keras.layers.Input([DIM, DIM, 3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(x)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    input_pretrained_model = tf.keras.Model(
        inputs=[i], outputs=[x], name="input_pretrained_model")

    base_model = tf.keras.applications.VGG16(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # freeze the first 15 layers of the base model. All other layers are trainable.
    for layer in base_model.layers[0:15]:
        layer.trainable = False

    # for idx, layer in enumerate(base_model.layers):
        #print("layer", idx + 1, ":", layer.name, "is trainable:", layer.trainable)

    # Create a new sequentail model and add the pretrained model
    model = tf.keras.models.Sequential()

    # Add the input for the pretrained model
    model.add(input_pretrained_model)

    # Add the pretrained model
    model.add(base_model)

    # Add a flatten layer to prepare the output of the cnn layer for the next layers
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(32, activation='relu'))

    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid'))

    return model
