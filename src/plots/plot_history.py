import json
import matplotlib.pyplot as plt


def plot_history(history, timestamp, save_output):
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

    if save_output:
        plt.savefig("./" + timestamp + "-history.png")
