
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def create_model(num_classes, verbose_level, save_output, timestamp):
    print("create model")
    vgg_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = vgg_model.get_layer("block5_pool")
    pretrained_model = Model(inputs=vgg_model.input,
                             outputs=transfer_layer.output)

    for layer in pretrained_model.layers[0:17]:
        layer.trainable = False

    for idx, layer in enumerate(pretrained_model.layers):
        print("layer", idx + 1, ":", layer.name,
              "is trainable:", layer.trainable)

    # Create a new sequentail model and add the pretrained model
    model = Sequential()

    # Add the pretrained model
    model.add(pretrained_model)

    # Add a flatten layer to prepare the output of the cnn layer for the next layers
    model.add(layers.Flatten())

    # Add a dense (aka. fully-connected) layer.
    # Add a dropout-layer which may prevent overfitting and improve generalization ability to unseen data.
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, activation='relu'))

    # Use the Sigmoid activation function for binary predictions, softmax for n-classes
    # We use the softmax function, because we have two classes (target_0 & target_1)
    model.add(layers.Dense(num_classes, activation='softmax'))

    # print model summary
    model.summary()

    # model callbacks
    callback_list = []

    # if the model does not improve for 10 epochs, stop the training
    stop_early = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    callback_list.append(stop_early)

    # if the output of the model should be saved, create a checkpoint callback function
    if save_output:
        # set the weight path for saving the model
        weight_path = "./" + timestamp + "-model.hdf5"
        # create the model checkpoint callback to save the model weights to a file
        checkpoint = ModelCheckpoint(
            weight_path,
            save_weights_only=True,
            verbose=verbose_level,
            save_best_only=True,
            monitor='val_loss',
            overwrite=True,
            mode='auto',
        )
        # append the checkpoint callback to the callback list
        callback_list.append(checkpoint)

    return model, callback_list
