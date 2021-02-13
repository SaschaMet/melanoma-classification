import os
import random
import numpy as np
import pandas as pd
from pathlib import Path


def check_image_paths(file_path):
    """ Helper function to validate the image paths

    Parameters:
        file_path (string): Path to the image

    Returns:
        The file path if the file exists, otherwise false if the file does not exist

    """
    img_file = Path(file_path)
    if img_file.is_file():
        return file_path
    return False


def get_train_data(base_path, image_path, image_type):
    """ Helper function to get the train dataset
    """
    # read the data from the train.csv file
    train = pd.read_csv(os.path.join(base_path, 'train.csv'))
    # add the image_path to the train set
    train['image_path'] = train['image_name'].apply(
        lambda x: image_path + "/train/" + x + image_type)
    # check if the we have an image
    train['image_path'] = train.apply(
        lambda row: check_image_paths(row['image_path']), axis=1)
    # if we do not have an image we will not include the data
    train = train[train['image_path'] != False]
    print("valid rows in train", train.shape[0])
    return train


def get_test_data(base_path, image_path, image_type):
    """ Helper function to get the test dataset
    """
    test = pd.read_csv(os.path.join(base_path, 'test.csv'))
    test['image_path'] = test['image_name'].apply(
        lambda x: image_path + "/test/" + x + image_type)
    test['image_path'] = test.apply(
        lambda row: check_image_paths(row['image_path']), axis=1)
    test = test[test['image_path'] != False]
    print("valid rows in test", test.shape[0])
    return test


def prepare_data(train, test):
    # getting dummy variables for gender
    sex_dummies = pd.get_dummies(train['sex'], prefix='sex', dtype="int")
    train = pd.concat([train, sex_dummies], axis=1)

    sex_dummies = pd.get_dummies(test['sex'], prefix='sex', dtype="int")
    test = pd.concat([test, sex_dummies], axis=1)

    # getting dummy variables for anatom_site_general_challenge
    anatom_dummies = pd.get_dummies(
        train['anatom_site_general_challenge'], prefix='anatom', dtype="int")
    train = pd.concat([train, anatom_dummies], axis=1)

    anatom_dummies = pd.get_dummies(
        test['anatom_site_general_challenge'], prefix='anatom', dtype="int")
    test = pd.concat([test, anatom_dummies], axis=1)

    # getting dummy variables for target column
    target_dummies = pd.get_dummies(
        train['target'], prefix='target', dtype="int")
    train = pd.concat([train, target_dummies], axis=1)

    # dropping not useful columns
    train.drop(['sex', 'diagnosis', 'benign_malignant',
                'anatom_site_general_challenge'], axis=1, inplace=True)
    test.drop(['sex', 'anatom_site_general_challenge'], axis=1, inplace=True)

    # replace missing age values wiht the mean age
    train['age_approx'] = train['age_approx'].fillna(
        int(np.mean(train['age_approx'])))
    test['age_approx'] = test['age_approx'].fillna(
        int(np.mean(test['age_approx'])))

    # convert age to int
    train['age_approx'] = train['age_approx'].astype('int')
    test['age_approx'] = test['age_approx'].astype('int')

    print("rows in train", train.shape[0])
    print("rows in test", test.shape[0])

    return train, test


def get_datasets(base_path, image_path, image_type):
    print("get train data")
    train = get_train_data(base_path, image_path, image_type)

    print("get test data")
    test = get_test_data(base_path, image_path, image_type)

    print("preparing datasets")
    train_set, test_set = prepare_data(train, test)

    return train_set, test_set


def balance_dataset(df, balance=1):
    # Balance the dataset¶
    # 1 means 50 / 50 => equal amount of positive and negative cases in Training
    # 4 = 20%; 8 = ~11%; 12 = ~8%

    p_inds = df[df.target == 1].index.tolist()
    np_inds = df[df.target == 0].index.tolist()

    np_sample = random.sample(np_inds, balance * len(p_inds))
    df = df.loc[p_inds + np_sample]
    print("Samples in df", df['target'].sum()/len(df))
    print("Remaining rows in df set", len(df))
    return df
