import matplotlib.pyplot as plt


def plot_metrics(history, timestamp, save_output):
    metrics = ['loss', 'accuracy', 'auc']
    plt.figure(figsize=(10, 8))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 1, n+1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch,
                 history.history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
    if save_output:
        plt.savefig("./" + timestamp + "-history.png")
