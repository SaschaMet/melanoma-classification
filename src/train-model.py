import os
import random
import warnings
import numpy as np
import tensorflow as tf
from datetime import datetime, date

from model.evaluation import evaluate_model
from model.create_model import create_model
from utils.create_splits import create_splits
from model.model_callbacks import get_model_callbacks
from utils.get_data import get_datasets, balance_dataset
from data.get_tf_records import get_dataset, verify_tf_records
from utils.data_generator import get_training_gen, get_validation_gen


SEED = 1
VERBOSE_LEVEL = 2

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


AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')


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
    batch_size = 32
    learning_rate = 1e-4
    optimizer = Adam(lr=learning_rate)
    loss = 'binary_crossentropy'
    metrics = [
        'accuracy',
        'AUC'
    ]

    # config depending on data
    balance_dataset = True

    num_classes = 2
    img_size = (1024, 1024)
    img_shape = (1024, 1024, 3)


main()
