import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_corr_plot(df):
    f, ax = plt.subplots(figsize=(10, 10))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)


def show_regplot_plot(df, x, y):
    sns.regplot(x=x, y=y, data=df)
