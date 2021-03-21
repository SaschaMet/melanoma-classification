import os
import sys
import random
import warnings
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime, date
from tensorflow.keras import backend as K

from data.load_tf_records import count_data_items, get_training_dataset, get_validation_dataset
from data.data_validation import get_class_distribution_of_dataset
from data.verify_tf_records import display_batch_of_images
from model.clr_callback import get_lr_callback, plot_clr
from model.evaluation import evaluate_model
from model.create_model import create_model

sys.path.append(str(Path('.').absolute().parent))
sys.path.insert(0, '/content/melanoma-classification')
sys.path.insert(1, '/content/melanoma-classification/src')

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


def get_model_parameters(lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = 'binary_crossentropy'
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    return loss, metrics, optimizer


def main():
    DIM = 768
    DIM_RESIZE = 512

    EPOCHS = 50
    BATCH_SIZE = 16
    NUM_CLASSES = 1
    SAVE_OUTPUT = True

    LR_MAX = 1e-4
    LR_MIN = 1e-6
    LR_START = 1e-5

    if tpu:
        BATCH_SIZE = 1024  # increase the batch size if we have a tpu
        # disable saving the outputs and tb because we do not have access to localhost on a tpu
        SAVE_OUTPUT = False

    # Set needed env variables based on the global variables
    os.environ["DIM"] = str(DIM)
    os.environ["BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["DIM_RESIZE"] = str(DIM_RESIZE)

    # Get the data
    MALIGNANT_IMAGES = tf.io.gfile.glob(
        'gs://kds-dd07414d960d14a2a8849e3ab696a4cb3299162b439e521ac886f393/*.tfrec')
    print("MALIGNANT TF Records", len(MALIGNANT_IMAGES))

    # This files include all malginant images from 2020, 2019, 2018 and 2017 competition as well as data directly from ISIC
    # Source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/169139
    MALIGNANT_IMAGES_V2 = tf.io.gfile.glob(
        'gs://kds-77aee4ba270735bf8687a77209581d50b8710ff6486f134f19b29759/*train*.tfrec')[15:]
    print("MALIGNANT TF Records V2", len(MALIGNANT_IMAGES_V2))

    BENIGN_IMAGES = tf.io.gfile.glob(
        'gs://kds-d499b3e548bc098dfea46bcac127fb30d0e465c2713c8457ff05deff/*.tfrec')
    print("BENIGN TF Records", len(BENIGN_IMAGES))

    # use only the first 160 batches for training
    BENIGN_IMAGES_TRAIN = BENIGN_IMAGES[0:160]

    TEST_FILENAMES = tf.io.gfile.glob(
        "gs://kds-3e36e1551f5588596f9fcb50129d35830a155849a24ad4825763d528/*.tfrec")
    print("TEST TF Records", len(TEST_FILENAMES))

    # Create the Training and Validation Dataset
    print(" ")

    # Training
    TRAINING_FILENAMES = [MALIGNANT_IMAGES[1], MALIGNANT_IMAGES[0]]
    TRAINING_FILENAMES = TRAINING_FILENAMES + \
        random.sample(MALIGNANT_IMAGES_V2, 20)
    TRAINING_FILENAMES = TRAINING_FILENAMES + \
        random.sample(BENIGN_IMAGES_TRAIN, len(TRAINING_FILENAMES))
    np.random.shuffle(TRAINING_FILENAMES)

    # Validation
    VALIDATION_FILENAMES = [
        MALIGNANT_IMAGES[2],
        BENIGN_IMAGES[162]
    ]

    TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
    VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

    print("TRAINING FILES", len(TRAINING_FILENAMES))
    print("VALODATION FILES", len(VALIDATION_FILENAMES))
    print(" ")

    print("TRAINING_IMAGES", TRAINING_IMAGES)
    print("VALIDATION_IMAGES", VALIDATION_IMAGES)

    training_dataset = get_training_dataset(
        TRAINING_FILENAMES, TRAINING_IMAGES, augment=True)
    validation_dataset = get_validation_dataset(
        VALIDATION_FILENAMES, repeat=True)

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

    example_batch = iter(training_dataset)
    initial_bias, class_weight = get_class_distribution_of_dataset(
        example_batch, TRAINING_IMAGES)

    # Clear the session - this helps when we are creating multiple models
    K.clear_session()

    # Creating the model in the strategy scope places the model on the TPU
    with strategy.scope():
        loss, metrics, optimizer = get_model_parameters(LR_START)
        model = create_model(NUM_CLASSES, DIM, initial_bias)

        if tpu:
            model.compile(
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                # Reduce python overhead, and maximize the performance of your TPU
                # Anything between 2 and `steps_per_epoch` could help here.
                steps_per_execution=steps_per_epoch / 10,
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
    print("Initial Training on 10 Epochs")
    print(" ")

    history = model.fit(
        training_dataset,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps_per_epoch,
        class_weight=class_weight,
        verbose=VERBOSE_LEVEL
    )

    print(" ")
    print("Start Fine tuning")
    print(" ")

    K.clear_session()
    with strategy.scope():
        model.trainable = True
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

    mode = 'triangular'
    step_size = 3.
    clr_callback = get_lr_callback(mode, LR_MIN, LR_MAX, step_size)
    plot_clr(mode, LR_MIN, LR_MAX, step_size, EPOCHS)

    # callbacks = get_model_callbacks(VERBOSE_LEVEL, SAVE_OUTPUT, timestamp)
    clr_callback = get_lr_callback(mode, LR_MIN, LR_MAX, step_size)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True)

    callbacks = [clr_callback, early_stopping_cb]

    history = model.fit(
        training_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps_per_epoch,
        class_weight=class_weight,
        verbose=VERBOSE_LEVEL
    )

    print(" ")
    print("Start evaluation process")
    print(" ")
    example_validation_dataset = get_validation_dataset(
        VALIDATION_FILENAMES, BATCH_SIZE)
    predictions, _, threshold = evaluate_model(model, example_validation_dataset, history,
                                               SAVE_OUTPUT, timestamp)
    predictions_mapped = [0 if x < threshold else 1 for x in predictions]

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
