import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_precision_recall_curve(labels, predictions):
    """ Helper function to plot the precision recall curve

    Parameters:
        precision
        recall

    Returns:
        Null
    """
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(precision, recall)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
