<<<<<<< HEAD
<<<<<<< HEAD
=======
import os

# configs to supress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

<<<<<<< HEAD

import json
import random
import warnings
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from datetime import datetime, date
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

=======
>>>>>>> 0e8338fc4f97ec9a3163a3f55ed0c78441ab21c2
=======
>>>>>>> 038190bd59383ad42659768eb2d7d399015d7e9c
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from datetime import datetime, date
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import warnings
import random
import json
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 038190bd59383ad42659768eb2d7d399015d7e9c
import os

# configs to supress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
<<<<<<< HEAD
=======
>>>>>>> 11433fbfc6a21bf2110886fa9a40ee873b6ccedb

>>>>>>> 0e8338fc4f97ec9a3163a3f55ed0c78441ab21c2
=======


>>>>>>> 038190bd59383ad42659768eb2d7d399015d7e9c
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)

SEED = 1
NUM_CLASSES = 2
VERBOSE_LEVEL = 2
SAVE_OUTPUT = True
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
OPTIMIZER = Adam(lr=LEARNING_RATE)
LOSS = 'binary_crossentropy'
METRICS = [
    'accuracy',
    'AUC'
]

CWD = os.getcwd()
IMAGE_TYPE = ".jpg"
BASE_PATH = '/kaggle/input/siim-isic-melanoma-classification'
PATH_TO_IMAGES = '/kaggle/input/siim-isic-melanoma-classification/jpeg'

MIXED_PRECISION = True
XLA_ACCELERATE = True
GPUS = 0

warnings.filterwarnings('ignore')
print("Tensorflow version " + tf.__version__)

GOOGLE_COLAB = False
if CWD == "/content":
    GOOGLE_COLAB = True
    print('Running in colab:', GOOGLE_COLAB)


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

# check on which system we are
if os.path.exists(CWD + '/data'):
    BASE_PATH = os.path.join(CWD, 'data')
    PATH_TO_IMAGES = BASE_PATH
    IMAGE_TYPE = ".png"
    print("change BASE_PATH to ", BASE_PATH)

elif GOOGLE_COLAB:
    CWD = "/content/melanoma-classification"
    BASE_PATH = os.path.join(CWD, 'data')
    PATH_TO_IMAGES = BASE_PATH
    IMAGE_TYPE = ".png"
    print("change BASE_PATH to ", BASE_PATH)


def seed_all(seed):
    """ Helper function to seed everything for getting reproducible results
    """
    print("Set seed")
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_KERAS'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)
    tf.random.set_seed(seed)


def check_image(file_path):
    """ Helper function to validate the image paths

    Parameters:
        file_path (string): Path to the image

    Returns:
        The file path if the file exists, otherwise false if the file does not exist

    """
    img_file = Path(file_path)
    if img_file.is_file():
        return file_path
    return False


def get_train_data():
    """ Helper function to get the train dataset
    """
    # read the data from the train.csv file
    train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    # add the image_path to the train set
    train['image_path'] = train['image_name'].apply(
        lambda x: PATH_TO_IMAGES + "/train/" + x + IMAGE_TYPE)
    # check if the we have an image
    train['image_path'] = train.apply(
        lambda row: check_image(row['image_path']), axis=1)
    # if we do not have an image we will not include the data
    train = train[train['image_path'] != False]
    print("valid rows in train", train.shape[0])
    return train


def get_test_data():
    """ Helper function to get the test dataset
    """
    test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    test['image_path'] = test['image_name'].apply(
        lambda x: PATH_TO_IMAGES + "/test/" + x + IMAGE_TYPE)
    test['image_path'] = test.apply(
        lambda row: check_image(row['image_path']), axis=1)
    test = test[test['image_path'] != False]
    print("valid rows in test", test.shape[0])
    return test


def create_splits(df, test_size, classToPredict):
    """ Helper function to create a train and a validation dataset

    Parameters:
    df (dataframe): The dataframe to split
    test_size (int): Size of the validation set
    classToPredict: The target column

    Returns:
    train_data (dataframe)
    val_data (dataframe)
    """
    train_data, val_data = train_test_split(
        df,  test_size=test_size, random_state=SEED, stratify=df[classToPredict])
    return train_data, val_data


def save_history(history, timestamp):
    """ Helper function to plot the history of a tensorflow model

        Parameters:
            history (history object): The history from a tf model
            timestamp (string): The timestamp of the function execution

        Returns:
            Null
    """
    f = plt.figure()
    f.set_figwidth(15)

    f.add_subplot(1, 2, 1)
    plt.plot(history['val_loss'], label='val loss')
    plt.plot(history['loss'], label='train loss')
    plt.legend()
    plt.title("Modell Loss")

    f.add_subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.plot(history['accuracy'], label='train accuracy')
    plt.legend()
    plt.title("Modell Accuracy")

    if SAVE_OUTPUT:
        plt.savefig("./" + timestamp + "-history.png")
        with open("./" + timestamp + "-history.json", 'w') as f:
            json.dump(history, f)


def plot_auc(t_y, p_y, timestamp):
    """ Helper function to plot the auc curve

    Parameters:
        t_y (array): True binary labels
        p_y (array): Target scores

    Returns:
        Null
    """
    fpr, tpr, thresholds = roc_curve(t_y, p_y, pos_label=1)
    fig, c_ax = plt.subplots(1, 1, figsize=(8, 8))
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % ('Target', auc(fpr, tpr)))
    c_ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.savefig("./" + timestamp + "-auc.png")


def calc_f1(prec, recall):
    """ Helper function to calculate the F1 Score

        Parameters:
            prec (int): precision
            recall (int): recall

        Returns:
            f1 score (int)
    """
    return 2*(prec*recall)/(prec+recall) if recall and prec else 0


def pred_to_binary(pred):
    """ Helper function turn the model predictions into a binary (0,1) format

    Parameters:
        pred (float): Model prediction

    Returns:
        binary prediction (int)
    """
    if pred < threshold:
        return 0
    else:
        return 1


def plot_confusion_matrix(cm, labels, timestamp):
    """ Helper function to plot a confusion matrix

        Parameters:
            cm (confusion matrix)

        Returns:
            Null
    """
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=55)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("./" + timestamp + "-cm.png")


def get_training_gen(df):
    """ Factory function to create a training image data generator

    Parameters:
        df (dataframe): Training dataframe

    Returns:
        Image Data Generator function
    """
    # prepare images for training
    train_idg = ImageDataGenerator(
        rescale=1 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.15,
        width_shift_range=0.15,
        shear_range=0.15,
        rotation_range=90,
        zoom_range=0.20,
        fill_mode='nearest'
    )

    train_gen = train_idg.flow_from_dataframe(
        seed=SEED,
        dataframe=df,
        directory=None,
        x_col='image_path',
        y_col=['target_0', 'target_1'],
        class_mode='raw',
        shuffle=True,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validate_filenames=False
    )

    return train_gen


def get_validation_gen(df):
    """ Factory function to create a validation image data generator

    Parameters:
        df (dataframe): Validation dataframe

    Returns:
        Image Data Generator function
    """
    # prepare images for validation
    val_idg = ImageDataGenerator(rescale=1. / 255.0)
    val_gen = val_idg.flow_from_dataframe(
        seed=SEED,
        dataframe=df,
        directory=None,
        x_col='image_path',
        y_col=['target_0', 'target_1'],
        class_mode='raw',
        shuffle=False,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validate_filenames=False
    )

    return val_gen


def load_pretrained_model():
    """ Helper function which returns a VGG16 model
    """
    base_model = VGG16(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    # freeze the first 15 layers of the base model. All other layers are trainable.
    for layer in base_model.layers[0:15]:
        layer.trainable = False

    for idx, layer in enumerate(base_model.layers):
        print("layer", idx + 1, ":", layer.name,
              "is trainable:", layer.trainable)

    return base_model


def create_model():
    """ Helper function which returns a tensorflow model
    """
    print("create model")

    # Create a new sequentail model and add the pretrained model
    model = Sequential()

    # Add the pretrained model
    model.add(load_pretrained_model())

    # Add a flatten layer to prepare the ouput of the cnn layer for the next layers
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
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    return model


seed_all(SEED)
train = get_train_data()
test = get_test_data()

# getting dummy variables for gender
sex_dummies = pd.get_dummies(train['sex'], prefix='sex', dtype="int")
train = pd.concat([train, sex_dummies], axis=1)

sex_dummies = pd.get_dummies(test['sex'], prefix='sex', dtype="int")
test = pd.concat([test, sex_dummies], axis=1)

# getting dummy variables for anatom_site_general_challenge
anatom_dummies = pd.get_dummies(
    train['anatom_site_general_challenge'], prefix='anatom', dtype="int")
train = pd.concat([train, anatom_dummies], axis=1)

anatom_dummies = pd.get_dummies(
    test['anatom_site_general_challenge'], prefix='anatom', dtype="int")
test = pd.concat([test, anatom_dummies], axis=1)

# getting dummy variables for target column
target_dummies = pd.get_dummies(train['target'], prefix='target', dtype="int")
train = pd.concat([train, target_dummies], axis=1)

# dropping not useful columns
train.drop(['sex', 'diagnosis', 'benign_malignant',
            'anatom_site_general_challenge'], axis=1, inplace=True)
test.drop(['sex', 'anatom_site_general_challenge'], axis=1, inplace=True)

# replace missing age values wiht the mean age
train['age_approx'] = train['age_approx'].fillna(
    int(np.mean(train['age_approx'])))
test['age_approx'] = test['age_approx'].fillna(
    int(np.mean(test['age_approx'])))

# convert age to int
train['age_approx'] = train['age_approx'].astype('int')
test['age_approx'] = test['age_approx'].astype('int')

print("rows in train", train.shape[0])
print("rows in test", test.shape[0])


# Balance the dataset¶
# 1 means 50 / 50 => equal amount of positive and negative cases in Training
# 4 = 20%; 8 = ~11%; 12 = ~8%
balance = 1
p_inds = train[train.target == 1].index.tolist()
np_inds = train[train.target == 0].index.tolist()

np_sample = random.sample(np_inds, balance * len(p_inds))
train = train.loc[p_inds + np_sample]
print("Samples in train", train['target'].sum()/len(train))
print("Remaining rows in train set", len(train))

model = create_model()
model.summary()

# get the current timestamp. This timestamp is used to save the model data with a unique name
now = datetime.now()
today = date.today()
current_time = now.strftime("%H:%M:%S")
timestamp = str(today) + "_" + str(current_time)


callback_list = []

# if the model does not improve for 10 epochs, stop the training
stop_early = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
callback_list.append(stop_early)

# if the output of the model should be saved, create a checkpoint callback function
if SAVE_OUTPUT:
    # set the weight path for saving the model
    weight_path = "./" + timestamp + "-model.hdf5"
    # create the model checkpoint callback to save the model wheights to a file
    checkpoint = ModelCheckpoint(
        weight_path,
        save_weights_only=True,
        verbose=VERBOSE_LEVEL,
        save_best_only=True,
        monitor='val_loss',
        overwrite=True,
        mode='auto',
    )
    # append the checkpoint callback to the callback list
    callback_list.append(checkpoint)


# create a training and validation dataset from the train df
train_df, val_df = create_splits(train, 0.2, 'target')

print("rows in train_df", train_df.shape[0])
print("rows in val_df", val_df.shape[0])

# because we do not need the target column anymore we can drop it
train_df.drop(['target'], axis=1, inplace=True)
val_df.drop(['target'], axis=1, inplace=True)

# call the generator functions
train_gen = get_training_gen(train_df)
val_gen = get_validation_gen(val_df)
valX, valY = val_gen.next()

model.compile(
    loss=LOSS,
    metrics=METRICS,
    optimizer=OPTIMIZER,
)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    verbose=VERBOSE_LEVEL,
    callbacks=callback_list,
    validation_data=(valX, valY),
)

print("Done training")

# plot model history
save_history(history.history, timestamp)

# plot the auc
y_t = []  # true labels
y_p = []  # predictions

# iterate over the validation df and make a prediction for each image
for i in tqdm(range(val_df.shape[0])):
    y_true = val_df.iloc[i].target_1
    image_path = val_df.iloc[i].image_path

    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img = keras.preprocessing.image.img_to_array(img)
    img = img / 255
    img_array = tf.expand_dims(img, 0)
    y_pred = model.predict(img_array)
    y_pred = tf.nn.softmax(y_pred)[0].numpy()[1]

    y_t.append(y_true)
    y_p.append(y_pred)

plot_auc(y_t, y_p, timestamp)

# calculate the precision, recall and the thresholds
precision, recall, thresholds = precision_recall_curve(y_t, y_p)

# calculate the f1 score
f1score = [calc_f1(precision[i], recall[i]) for i in range(len(thresholds))]

# get the index from the highest f1 score
idx = np.argmax(f1score)

# get the precision, recall, threshold and the f1score
precision = round(precision[idx], 4)
recall = round(recall[idx], 4)
threshold = round(thresholds[idx], 4)
f1score = round(f1score[idx], 4)

print('Precision:', precision)
print('Recall:', recall)
print('Threshold:', threshold)
print('F1 Score:', f1score)


y_pred_binary = [pred_to_binary(x) for x in y_p]

# create a confusion matrix
cm = confusion_matrix(y_t, y_pred_binary)

cm_plot_label = ['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label, timestamp)
<<<<<<< HEAD
<<<<<<< HEAD


metrics = {
    'f1score': str(f1score),
    'precision': str(precision),
    'recall': str(recall),
    'threshold': str(threshold),
}

with open('metrics.txt', 'w') as file:
    file.write(json.dumps(metrics))
=======
<<<<<<< HEAD
=======
>>>>>>> 038190bd59383ad42659768eb2d7d399015d7e9c


metrics = {
    'f1score': str(f1score),
    'precision': str(precision),
    'recall': str(recall),
    'threshold': str(threshold),
}

with open('metrics.txt', 'w') as file:
    file.write(json.dumps(metrics))
<<<<<<< HEAD
=======
>>>>>>> 0e8338fc4f97ec9a3163a3f55ed0c78441ab21c2
>>>>>>> 11433fbfc6a21bf2110886fa9a40ee873b6ccedb
=======
>>>>>>> 038190bd59383ad42659768eb2d7d399015d7e9c
