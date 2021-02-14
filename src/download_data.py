import os
import json
import shutil


def download_dataset():
    CWD = os.getcwd()

    print("Delete data dir")
    try:
        shutil.rmtree(CWD + "/data")
    except:
        pass

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

    # make data directory
    os.system('mkdir data')

    # go to data directory and download dataset
    os.system('cd data && kaggle datasets download -d cdeotte/jpeg-melanoma-768x768')

    # unzip the zip file
    os.system('cd data && unzip jpeg-melanoma-768x768.zip')

    # remove the not needed files
    os.system('cd data && rm jpeg-melanoma-768x768.zip')
    os.system('cd data && rm sample_submission.csv')


download_dataset()
