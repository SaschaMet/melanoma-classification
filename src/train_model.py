import os
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, date
from sklearn.model_selection import train_test_split

from model.create_model import create_model
from data.get_tf_records import get_dataset
from model.model_callbacks import get_model_callbacks


SEED = 1
VERBOSE_LEVEL = 1

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
print("Tensorflow version " + tf.__version__)
strategy = tf.distribute.get_strategy()
num_gpus = len(
    tf.config.experimental.list_physical_devices('GPU')
)

if num_gpus > 0:
    print("Num GPUs Available: ", num_gpus)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')


strategy = tf.distribute.get_strategy()
REPLICAS = strategy.num_replicas_in_sync
AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print("Using default strategy for CPU and single GPU")
print("REPLICAS:", REPLICAS)


def main():

    # global config
    save_output = True

    # check the runtime
    cwd = os.getcwd()
    if cwd == "/content":
        cwd = "/content/melanoma-classification"
        print('Running on google colab')

    epochs = 50
    batch_size = 128
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = [
        'accuracy',
        'AUC'
    ]

    dim = 512
    num_classes = 1
    img_shape = (dim, dim, 3)

    training_tfrecords, validation_tfrecords = train_test_split(
        tf.io.gfile.glob(cwd + "/data/train*.tfrec"),
        test_size=0.2,
        random_state=SEED
    )
    test_tfrecords = tf.io.gfile.glob(cwd + "/data/test*.tfrec")

    train_df = pd.read_csv(cwd + '/data/train.csv')
    malignant = np.count_nonzero(train_df['target'])
    total_img = train_df['target'].size
    benign = total_img - malignant

    bias = np.log([malignant/benign])

    weight_for_0 = (1 / benign)*(total_img)/2.0
    weight_for_1 = (1 / malignant)*(total_img)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    print("Train TFRecord Files:", len(training_tfrecords))
    print("Validation TFRecord Files:", len(validation_tfrecords))
    print("Test TFRecord Files:", len(test_tfrecords))

    files_train = count_data_items(training_tfrecords)
    files_val = count_data_items(validation_tfrecords)

    steps_per_epoch = files_train/batch_size//REPLICAS
    validation_steps = files_val/batch_size//REPLICAS

    print("steps_per_epoch", steps_per_epoch)
    print("validation_steps", validation_steps)

    train_ds = get_dataset(training_tfrecords, augment=True,
                           shuffle=True, repeat=False, dim=dim, batch_size=batch_size)
    val_ds = get_dataset(validation_tfrecords, augment=False,
                         shuffle=False, repeat=False, dim=dim, batch_size=batch_size)

    with strategy.scope():

        model = create_model(img_shape, num_classes, bias)

        model.compile(
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
        )

        model.fit(
            train_ds,
            epochs=epochs,
            callbacks=get_model_callbacks(
                VERBOSE_LEVEL, save_output, timestamp),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            verbose=VERBOSE_LEVEL,
            class_weight=class_weight
        )

        model.save('model_files')

        # print('Computing predictions...')
        # images_ds = val_ds.map(lambda image, idnum: image)
        # probabilities = model.predict(images_ds)
        # print("probabilities", probabilities)


main()
