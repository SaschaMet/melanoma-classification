# SIIM-ISIC Melanoma Classification
## Identify melanoma in lesion images

This repository holds the source code for a Tensorflow CNN to identify melanoma in lesion images.

You can find all the code in the `main.ipynb` file.

You can also find this notebook on Kaggle: https://www.kaggle.com/saschamet/siim-isic-melanoma-classification.
Note: This notebook's results may be different compared to the `main.ipynb` file due to the different data sources.


## Important notice for reproducing the results on google colab
Before running the notebook, you have to download the dataset from Kaggle. I prepared a script that will do just that. You can find it here: `nb_on_colab.txt`. This script will install the kaggle package, and it will connect to your kaggle account (this is necessary to download the dataset). After mounting your local google drive folder, you can either link your `kaggle.json` file or add your username and key inline. For detailed documentation, you can visit the kaggle page: https://www.kaggle.com/docs/api.


## Important notice for local development with vs code
To start developing with vs code, please install the "Remote - Containers" extension and start the project in a dev container. After the dev container has started (this may take some time), execute the install script with `chmod +x install.sh && ./install.sh`. This script will install all necessary dependencies.