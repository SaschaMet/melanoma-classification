import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet101V2


def get_pretrained_model(img_shape):
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        pretrained_model = ResNet101V2(
            include_top=False,
            weights="imagenet",
            input_tensor=Sequential([])(layers.Input(shape=img_shape)),
        )

    # We unfreeze some blocks while leaving BatchNorm layers frozen
    for idx, layer in enumerate(pretrained_model.layers):
        layer.trainable = False
        if not isinstance(layer, layers.BatchNormalization):
            if "conv5" in layer.name:
                layer.trainable = True
        print("layer", idx + 1, ":", layer.name,
              "is trainable:", layer.trainable)

    return pretrained_model


def create_model(img_shape, num_classes):
    print("create model")

    # create a new sequentail model
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = Sequential()

        # add the pretrained model
        pretrained_model = get_pretrained_model(img_shape)
        model.add(pretrained_model)

        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(num_classes, activation='softmax'))

        model.summary()

    return model
