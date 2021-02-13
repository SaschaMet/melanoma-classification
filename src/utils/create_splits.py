from sklearn.model_selection import train_test_split


def create_splits(df, test_size, classToPredict, seed):
    """ Helper function to create a train and a validation dataset

    Parameters:
    df (dataframe): The dataframe to split
    test_size (int): Size of the validation set
    classToPredict: The target column

    Returns:
    train_data (dataframe)
    val_data (dataframe)
    """
    train_data, val_data = train_test_split(
        df,  test_size=test_size, random_state=seed, stratify=df[classToPredict])

    # because we do not need the classToPredict column anymore we can drop it
    train_data.drop([classToPredict], axis=1, inplace=True)
    val_data.drop([classToPredict], axis=1, inplace=True)

    return train_data, val_data
