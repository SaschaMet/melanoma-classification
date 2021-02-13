import os
import random
import warnings
import numpy as np
import tensorflow as tf
from datetime import datetime, date

from model.evaluation import evaluate_model
from model.create_model import create_model
from utils.create_splits import create_splits
from utils.get_data import get_datasets, balance_dataset
from utils.data_generator import get_training_gen, get_validation_gen

SEED = 1
VERBOSE_LEVEL = 2

# Tensorflow
GPUS = 0
XLA_ACCELERATE = True
MIXED_PRECISION = True

# suppress tf logs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)

# seed everything
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_KERAS'] = str(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)


# get the current timestamp. This timestamp is used to save the model data with a unique name
now = datetime.now()
today = date.today()
current_time = now.strftime("%H:%M:%S")
timestamp = str(today) + "_" + str(current_time)


# Tensorflow execution optimizations
# Source: https://www.tensorflow.org/guide/mixed_precision & https://www.tensorflow.org/xla
GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
if GPUS == 0:
    DEVICE = 'CPU'
else:
    DEVICE = 'GPU'
    if MIXED_PRECISION:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print('Mixed precision enabled')
    if XLA_ACCELERATE:
        tf.config.optimizer.set_jit(True)
        print('Accelerated Linear Algebra enabled')


def get_data_for_training(base_path, image_path, image_type, balance_data, seed):
    train, test = get_datasets(base_path, image_path, image_type)

    if balance_data:
        train = balance_dataset(train)

    # create a training and validation dataset from the train df
    train_df, val_df = create_splits(train, 0.2, 'target', seed)

    print("rows in train_df", train_df.shape[0])
    print("rows in val_df", val_df.shape[0])

    return train_df, val_df


def train_model(train_df, val_df, loss, metrics, optimizer, epochs, batch_size, img_size, num_classes, save_output, seed):

    # call the generator functions
    train_gen = get_training_gen(train_df, seed, img_size, batch_size)
    val_gen = get_validation_gen(val_df, seed, img_size, batch_size)
    valX, valY = val_gen.next()

    model, callback_list = create_model(
        num_classes,
        VERBOSE_LEVEL,
        save_output,
        timestamp
    )

    model.compile(
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=VERBOSE_LEVEL,
        callbacks=callback_list,
        validation_data=(valX, valY),
    )

    print("Done with training")

    print("Start evaluating")
    evaluate_model(model, val_df, history, timestamp, img_size)

    print("Done")


def main():
    from tensorflow.keras.optimizers import Adam

    # global config
    save_output = True

    # check the runtime
    CWD = os.getcwd()
    if CWD == "/content":
        CWD = "/content/melanoma-classification"
        print('Running on google colab')

    base_path = os.path.join(CWD, 'data')
    path_to_images = os.path.join(CWD, 'data')

    epochs = 40
    batch_size = 64
    learning_rate = 1e-4
    optimizer = Adam(lr=learning_rate)
    loss = 'binary_crossentropy'
    metrics = [
        'accuracy',
        'AUC'
    ]

    # config depending on data
    balance_dataset = True
    img_size = (224, 224)
    image_type = ".png"
    num_classes = 2

    train_df, val_df = get_data_for_training(
        base_path,
        path_to_images,
        image_type,
        balance_dataset,
        SEED
    )

    train_model(
        train_df,
        val_df,
        loss,
        metrics,
        optimizer,
        epochs,
        batch_size,
        img_size,
        num_classes,
        save_output,
        SEED
    )


main()
