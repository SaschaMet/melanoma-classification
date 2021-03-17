import os
import sys
import random
import warnings
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow_addons as tfa
from datetime import datetime, date
from tensorflow.keras import backend as K

from data.load_tf_records import count_data_items, get_training_dataset, get_validation_dataset, get_prediction_validation_dataset
from data.verify_tf_records import display_batch_of_images
from model.model_callbacks import get_model_callbacks
from model.evaluation import evaluate_model
from model.create_model import create_model
from model.clr_callback import plot_clr

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


def get_model_parameters(steps_per_epoch, BASE_LR, epochs):
    optimizer = tfa.optimizers.RectifiedAdam(
        total_steps=int(steps_per_epoch * epochs),
        warmup_proportion=0.15,
        min_lr=BASE_LR,
    )
    loss = 'binary_crossentropy'
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    return loss, metrics, optimizer


def main():
    DIM = 768
    EPOCHS = 50
    BATCH_SIZE = 16
    NUM_CLASSES = 1
    SAVE_OUTPUT = True
    USE_TENSORBOARD = True

    BASE_LR = 1e-6
    MAX_LR = 1e-3

    if tpu:
        BATCH_SIZE = 1024  # increase the batch size if we have a tpu
        # disable saving the outputs and tb because we do not have access to localhost on a tpu
        SAVE_OUTPUT = False
        USE_TENSORBOARD = False

    # Set needed env variables based on the global variables
    os.environ["DIM"] = str(DIM)
    os.environ["BATCH_SIZE"] = str(BATCH_SIZE)

    # Get the data
    MALIGNANT_IMAGES = tf.io.gfile.glob(os.environ["PATH_TO_MALIGNANT_FILES"])
    BENIGN_IMAGES = tf.io.gfile.glob(os.environ["PATH_TO_BENIGN_FILES"])
    TEST_FILENAMES = tf.io.gfile.glob(os.environ["PATH_TO_TEST_FILES"])

    print('There are', count_data_items(MALIGNANT_IMAGES),
          'malignant images in', len(MALIGNANT_IMAGES), 'files')

    print('There are', count_data_items(BENIGN_IMAGES),
          'benign images in', len(BENIGN_IMAGES), 'files')

    print('There are', count_data_items(TEST_FILENAMES),
          'test images in', len(TEST_FILENAMES), 'files')

    # Create the Training and Validation Dataset
    print("Creating datasets...")
    TRAINING_FILENAMES = [
        MALIGNANT_IMAGES[0],
        MALIGNANT_IMAGES[1],
        BENIGN_IMAGES[0],
        BENIGN_IMAGES[1]
    ]

    VALIDATION_FILENAMES = [
        MALIGNANT_IMAGES[2],
        BENIGN_IMAGES[39]
    ]

    print('There are', count_data_items(TRAINING_FILENAMES), 'training images')
    print('There are', count_data_items(
        VALIDATION_FILENAMES), 'validation images')

    training_dataset = get_training_dataset(TRAINING_FILENAMES, augment=True)
    validation_dataset = get_validation_dataset(VALIDATION_FILENAMES)

    steps_per_epoch = count_data_items(
        TRAINING_FILENAMES) // BATCH_SIZE * REPLICAS
    validation_steps_per_epoch = count_data_items(
        VALIDATION_FILENAMES) // BATCH_SIZE * REPLICAS

    print("Epochs", EPOCHS)
    print("BATCH SIZE", BATCH_SIZE)
    print("steps_per_epoch", steps_per_epoch)
    print("validation_steps_per_epoch", validation_steps_per_epoch)

    # get the current timestamp. This timestamp is used to save the model data with a unique name
    now = datetime.now()
    today = date.today()
    current_time = now.strftime("%H:%M:%S")
    timestamp = str(today) + "_" + str(current_time)

    # get the model callbacks
    callbacks = get_model_callbacks(
        steps_per_epoch, BASE_LR, MAX_LR, VERBOSE_LEVEL, SAVE_OUTPUT, timestamp, USE_TENSORBOARD)

    # Clear the session - this helps when we are creating multiple models
    K.clear_session()

    # Creating the model in the strategy scope places the model on the TPU
    with strategy.scope():

        loss, metrics, optimizer = get_model_parameters(
            steps_per_epoch, BASE_LR, EPOCHS)

        model = create_model(NUM_CLASSES, DIM)

        if tpu:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                # Reduce python overhead, and maximize the performance of your TPU
                # Anything between 2 and `steps_per_epoch` could help here.
                steps_per_execution=steps_per_epoch / 4,
            )
        else:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
            )

    print(" ")
    print(model.summary())
    print(" ")
    print("Fit model")
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

    print(" ")
    print("Start evaluation process")
    print(" ")
    example_validation_dataset = get_prediction_validation_dataset(
        VALIDATION_FILENAMES, BATCH_SIZE)
    predictions, _, threshold = evaluate_model(model, example_validation_dataset, history,
                                               SAVE_OUTPUT, timestamp)
    predictions_mapped = [0 if x < threshold else 1 for x in predictions]

    plot_clr(callbacks[0])

    example_validation_dataset = example_validation_dataset.unbatch().batch(20)
    example_validation_batch = iter(example_validation_dataset)
    validation_image_batch, validation_label_batch = next(
        example_validation_batch)
    display_batch_of_images(
        (validation_image_batch, validation_label_batch), predictions_mapped)

    print(" ")
    print("Done")
    print(" ")


main()
