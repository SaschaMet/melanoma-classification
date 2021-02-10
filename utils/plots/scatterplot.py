import seaborn as sns
import matplotlib.pyplot as plt


def show_scatterplot(df, x, y):
    sns.set(color_codes=True)
    ax = sns.scatterplot(x=x, y=y, data=df)
    plt.show()
