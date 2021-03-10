import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, labels, timestamp, save_output):
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
                 color="white" if cm[i, j] < thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if save_output:
        plt.savefig("./" + timestamp + "-cm.png")
