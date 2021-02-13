import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
