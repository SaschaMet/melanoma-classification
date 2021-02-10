import seaborn as sns


def show_bar_plot(df, x, y):
    ax = sns.barplot(x=x, y=y, data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
