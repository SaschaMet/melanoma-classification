import os
import json


def download_dataset():
    print("Download data")

    # check were the kaggle auth file is stored
    path_to_auth_file = "kaggle.json"
    if not os.path.exists(path_to_auth_file):
        path_to_auth_file = '/content/drive/MyDrive/Colab Notebooks/_auth/kaggle.json'

    # read the kaggle.json file
    with open(path_to_auth_file) as json_file:
        data = json.load(json_file)
        os.environ['KAGGLE_USERNAME'] = data["username"]
        os.environ['KAGGLE_KEY'] = data["key"]

    os.system('mkdir data')

    # go to data directory and download dataset
    os.system(
        'cd data && kaggle datasets download -d cdeotte/melanoma-512x512')

    # unzip the zip file
    os.system('cd data && unzip -o melanoma-512x512.zip')

    # remove the not needed files
    os.system('cd data && rm melanoma-512x512.zip')


download_dataset()
