import sys  # isort:skip
import json  # isort:skip
import numpy as np  # isort:skip
from tqdm import tqdm  # isort:skip
import tensorflow as tf  # isort:skip
from tensorflow import keras  # isort:skip
from sklearn.metrics import precision_recall_curve, confusion_matrix  # isort:skip


sys.path.insert(0, '..')  # isort:skip


def calc_f1(prec, recall):
    """ Helper function to calculate the F1 Score

        Parameters:
            prec (int): precision
            recall (int): recall

        Returns:
            f1 score (int)
    """
    return 2*(prec*recall)/(prec+recall) if recall and prec else 0


def pred_to_binary(pred, threshold):
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


def evaluate_model(model, val_df, history, timestamp, img_size):
    from src.plots.plot_auc import plot_auc  # isort:skip
    from src.plots.plot_history import plot_history  # isort:skip
    from src.plots.plot_confusion_matrix import plot_confusion_matrix  # isort:skip

    y_t = []  # true labels
    y_p = []  # predictions

    # iterate over the validation df and make a prediction for each image
    for i in tqdm(range(val_df.shape[0])):
        y_true = val_df.iloc[i].target_1
        image_path = val_df.iloc[i].image_path

        img = keras.preprocessing.image.load_img(
            image_path, target_size=img_size)
        img = keras.preprocessing.image.img_to_array(img)
        img = img / 255
        img_array = tf.expand_dims(img, 0)
        y_pred = model.predict(img_array)
        y_pred = tf.nn.softmax(y_pred)[0].numpy()[1]

        y_t.append(y_true)
        y_p.append(y_pred)

    # calculate the precision, recall and the thresholds
    precision, recall, thresholds = precision_recall_curve(y_t, y_p)

    # calculate the f1 score
    f1score = [calc_f1(precision[i], recall[i])
               for i in range(len(thresholds))]

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

    # save the metrics
    metrics = {
        'f1score': str(f1score),
        'precision': str(precision),
        'recall': str(recall),
        'threshold': str(threshold),
    }

    with open('metrics.txt', 'w') as file:
        file.write(json.dumps(metrics))

    # create a confusion matrix
    y_pred_binary = [pred_to_binary(x) for x in y_p]
    cm = confusion_matrix(y_t, y_pred_binary)

    cm_plot_label = ['benign', 'malignant']
    plot_confusion_matrix(cm, cm_plot_label, timestamp)

    # plot model history
    plot_history(history.history, timestamp)

    # plot auc
    plot_auc(y_t, y_p, timestamp)

    return y_t, y_p
