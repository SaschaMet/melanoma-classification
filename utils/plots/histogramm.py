import matplotlib.pyplot as plt


def plot_hist(data, bins="auto"):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins)
    plt.show()
