import seaborn as sns


def plot_hist(data, bins=10):
    sns.histplot(data, bins=bins, kde=True)
