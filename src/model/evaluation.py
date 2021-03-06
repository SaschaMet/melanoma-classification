import json
import math
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

from plots.plot_auc import plot_auc
from plots.plot_history import plot_history
from plots.plot_confusion_matrix import plot_confusion_matrix


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


def predict_on_dataset(model, dataset, number_of_images, test_batch_size=50):
    steps = math.ceil(number_of_images / test_batch_size)
    dataset = dataset.unbatch().batch(test_batch_size)

    _, labels = tuple(zip(*dataset))

    label_list = []
    for i in range(steps):
        for item in labels[i].numpy().flatten():
            # read the target variable as the label
            label_list.append(item)

    print("start predicting ...")

    dataset = dataset.map(lambda image, idnum: image)
    predictions = model.predict(dataset, steps=steps, verbose=1)
    predictions = predictions.flatten().tolist()
    predictions = [round(num, 4) for num in predictions]

    return predictions, label_list


def evaluate_model(model, dataset, history, number_of_images, save_output, timestamp):
    predictions, labels = predict_on_dataset(model, dataset, number_of_images)

    # calculate the precision, recall and the thresholds
    precision, recall, thresholds = precision_recall_curve(labels, predictions)

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

    if save_output:
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
    y_pred_binary = [pred_to_binary(x, threshold) for x in predictions]
    cm = confusion_matrix(labels, y_pred_binary)

    cm_plot_label = ['benign', 'malignant']
    plot_confusion_matrix(cm, cm_plot_label, timestamp, save_output)

    # plot model history
    plot_history(history.history, timestamp, save_output)

    # plot auc
    plot_auc(labels, predictions, timestamp, save_output)

    return predictions, labels, threshold
