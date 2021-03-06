import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from datetime import datetime, date
from tensorflow.keras import backend as K

from data.load_tf_records import count_data_items, get_training_dataset, get_validation_dataset
from data.validation import get_class_distribution_of_dataset
from data.verify_tf_records import display_batch_of_images
from model.model_callbacks import get_model_callbacks
from model.evaluation import evaluate_model

sys.path.append(str(Path('.').absolute().parent))

# Global Settings
SEED = 1
VERBOSE_LEVEL = 1

# suppress tf logs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(VERBOSE_LEVEL)

# seed everything
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# get the current timestamp. This timestamp is used to save the model data with a unique name
now = datetime.now()
today = date.today()
current_time = now.strftime("%H:%M:%S")
timestamp = str(today) + "_" + str(current_time)


# environment settings
print("Tensorflow version " + tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE

# Tensorflow execution optimizations
# Source: https://www.tensorflow.org/guide/mixed_precision & https://www.tensorflow.org/xla
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
except:
    pass

if tpu:
    print("Try connecting to a tpu")
    try:
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        # tf.config.experimental_connect_to_cluster(tpu)
        # tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        REPLICAS = strategy.num_replicas_in_sync
        print("REPLICAS:", REPLICAS)
    except ValueError as error:
        raise BaseException('An exception occurred: {}'.format(error))
    except BaseException as error:
        raise BaseException('An exception occurred: {}'.format(error))
else:
    num_gpus = len(
        tf.config.experimental.list_physical_devices('GPU')
    )
    print("Using default strategy for CPU and single GPU")
    print("Num GPUs Available: ", num_gpus)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')
    strategy = tf.distribute.get_strategy()
    REPLICAS = strategy.num_replicas_in_sync
    print("REPLICAS:", REPLICAS)


def get_model_metrics():
    loss = 'binary_crossentropy'
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    return loss, metrics


def main():
    DIM = 768
    EPOCHS = 50
    BATCH_SIZE = 4
    NUM_CLASSES = 1
    SAVE_OUTPUT = True
    USE_TENSORBOARD = True

    os.environ["DIM"] = str(DIM)

    if tpu:
        BATCH_SIZE = 128  # increase the batch size if we have a tpu
        # disable saving the outputs and tb because we do not have access to localhost on a tpu
        SAVE_OUTPUT = False
        USE_TENSORBOARD = False

    print("BATCH_SIZE", BATCH_SIZE)

    PATH_TO_TRAIN_FILES = os.environ["PATH_TO_TRAIN_FILES"]
    TRAINING_FILENAMES = tf.io.gfile.glob(
        PATH_TO_TRAIN_FILES + '/*train*.tfrec')
    VALIDATION_FILENAMES = tf.io.gfile.glob(
        PATH_TO_TRAIN_FILES + '/*val*.tfrec')

    print("Number of train files", len(TRAINING_FILENAMES))
    print("Number of validation files", len(VALIDATION_FILENAMES))

    PATH_TO_TEST_FILES = os.environ["PATH_TO_TEST_FILES"]
    TEST_FILENAMES = tf.io.gfile.glob(PATH_TO_TEST_FILES + '/*test*.tfrec')
    print("Number of test files", len(TEST_FILENAMES))

    print('There are', count_data_items(TRAINING_FILENAMES), 'train images')
    print('There are', count_data_items(VALIDATION_FILENAMES), 'val images')
    print('There are', count_data_items(TEST_FILENAMES), 'test images')

    training_dataset = get_training_dataset(
        TRAINING_FILENAMES, BATCH_SIZE, tpu)
    validation_dataset = get_validation_dataset(
        VALIDATION_FILENAMES, BATCH_SIZE, tpu)

    print("Calculate the class distribution")
    train_csv = pd.read_csv("data/train.csv")
    malignant_cases, benign_cases = get_class_distribution_of_dataset(
        train_csv)

    bias = np.log([malignant_cases/benign_cases])
    output_bias = tf.keras.initializers.Constant(bias)

    # Increase the steps per epoch by MULTIPLIER if we train on a TPU
    if tpu:
        MULTIPLIER = 20
        steps_per_epoch = (count_data_items(TRAINING_FILENAMES) /
                           BATCH_SIZE//REPLICAS) * MULTIPLIER
        validation_steps_per_epoch = (count_data_items(
            VALIDATION_FILENAMES)/BATCH_SIZE//REPLICAS) * MULTIPLIER
    else:
        steps_per_epoch = steps_per_epoch = (count_data_items(TRAINING_FILENAMES) /
                                             BATCH_SIZE//REPLICAS)
        validation_steps_per_epoch = (count_data_items(
            VALIDATION_FILENAMES)/BATCH_SIZE//REPLICAS)

    print("Train steps per epoch", steps_per_epoch)
    print("Validation steps per epoch", validation_steps_per_epoch)

    # get the current timestamp. This timestamp is used to save the model data with a unique name
    now = datetime.now()
    today = date.today()
    current_time = now.strftime("%H:%M:%S")
    timestamp = str(today) + "_" + str(current_time)

    # get the model callbacks
    callbacks = get_model_callbacks(
        strategy, EPOCHS, VERBOSE_LEVEL, SAVE_OUTPUT, timestamp, USE_TENSORBOARD)

    # Clear the session - this helps when we are creating multiple models
    K.clear_session()

    # Creating the model in the strategy scope places the model on the TPU
    with strategy.scope():

        print("Create a pretrained resnet model")

        # preprocessing needed for resnet
        i = tf.keras.layers.Input([DIM, DIM, 3], dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        input_pretrained_model = tf.keras.Model(
            inputs=[i], outputs=[x], name="input_pretrained_model")

        # create base model
        pretrained_model = tf.keras.applications.ResNet101V2(
            weights="imagenet",  # Load weights pre-trained on ImageNet
            input_shape=(DIM, DIM, 3),
            include_top=False,
        )

        # freeze the model
        # we can't use pretrained_model.trainable = False because of an issue between the keras and tf implementation
        # https://github.com/tensorflow/tensorflow/issues/29535
        for layer in pretrained_model.layers:
            layer.trainable = False

        # create the final model
        model = tf.keras.Sequential([
            input_pretrained_model,
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(
                NUM_CLASSES, activation='sigmoid', bias_initializer=output_bias)
        ])

        loss, metrics = get_model_metrics()

        if tpu:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer='adam',
                # Reduce python overhead, and maximize the performance of your TPU
                # Anything between 2 and `steps_per_epoch` could help here.
                steps_per_execution=steps_per_epoch / 10,
            )
        else:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer='adam',
            )

        print(" ")
        print(model.summary())
        print(" ")
        print("Fit model Nr. 1")
        print(" ")
        history = model.fit(
            training_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            validation_steps=validation_steps_per_epoch,
            verbose=VERBOSE_LEVEL
        )

        example_validation_dataset = get_validation_dataset(
            VALIDATION_FILENAMES, BATCH_SIZE, tpu)
        validation_images = count_data_items(VALIDATION_FILENAMES)
        evaluate_model(model, example_validation_dataset, history,
                       validation_images, SAVE_OUTPUT, timestamp)

    print(" ")
    print("Compile model Nr. 2")
    print(" ")

    K.clear_session()
    with strategy.scope():
        print("conv5_block: ")
        # We unfreeze the last conv block while leaving BatchNorm layers frozen
        for layer in model.layers[1].layers:
            if not isinstance(layer, layers.BatchNormalization) and "conv5_" in layer.name:
                layer.trainable = True
            if "conv5_" in layer.name:
                print(layer.name, layer.trainable)
        print(" ")

        loss, metrics = get_model_metrics()

        if tpu:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer='adam',
                # Reduce python overhead, and maximize the performance of your TPU
                # Anything between 2 and `steps_per_epoch` could help here.
                steps_per_execution=steps_per_epoch / 10,
            )
        else:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer='adam',
            )

    print(" ")
    print(model.summary())
    print(" ")
    print("Fit model Nr. 2")
    print(" ")

    history = model.fit(
        training_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps_per_epoch,
        verbose=VERBOSE_LEVEL
    )

    predictions, _, threshold = evaluate_model(model, example_validation_dataset, history,
                                               validation_images, SAVE_OUTPUT, timestamp)

    predictions_mapped = [0 if x < threshold else 1 for x in predictions]

    example_validation_dataset = example_validation_dataset.unbatch().batch(20)
    example_validation_batch = iter(example_validation_dataset)
    validation_image_batch, validation_label_batch = next(
        example_validation_batch)
    display_batch_of_images(
        (validation_image_batch, validation_label_batch), predictions_mapped)


main()
