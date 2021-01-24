import pandas as pd
from pathlib import Path


def check_image(file_path):
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


def get_data(path_to_csv, subset, img_dir):
    """Helper function to load a csv file as a pandas df and checks if the img exists

    Args:
        path_to_csv (str): Path to a csv file
        img_dir (str): Directory where images are stored

    Returns:
        Pandas Dataframe: A pandas dataframe with validated images
    """
    # read the data from the csv file
    df = pd.read_csv(path_to_csv)
    # add the image_path to the df
    df['image_path'] = df['image_name'].apply(
        lambda x: img_dir + subset + "/" + x + ".png")
    # check if the we have an image
    df['image_path'] = df.apply(
        lambda row: check_image(row['image_path']), axis=1)
    # if we do not have an image we will not include the data
    df = df[df['image_path'] != False]
    print("valid rows", df.shape[0])
    return df
