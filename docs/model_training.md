# Model Training

This guide will provide an overview of how to train a model. There are two ways to do this:

1. Use Kaggle.com
This is the simplest solution. Just upload this notebook to Kaggle.com and run it: `notebooks/melanoma-efficientnetb5-noisy-student.ipynb`
Note: This notebook can't be executed locally because it used datasets stored on Kaggle.

2. Train locally
Besides the notebook mentioned above, every other notebook can be used locally. However, there are a few things to keep in mind:
I used Google Colab for training the neural networks, so every notebook is created for this use case. If you want to run a notebook on a local PC, you will have to change the import paths in the "Setup" section to match your local setup. In addition to that, I used TPUs for training the models. The parameters and the data pre-processing steps are designed to run on this hardware. Depending on your setup, you will therefore have to adapt them.

Because you can't use the Datasets hosted on Kaggle, you have to store them on your own. This can be locally on your PC. However, if you want to use a TPU, you have to store them on Google Cloud because you can only use TPUs when the data is stored on the google cloud platform. The Datasets can be downloaded here:

- [https://melanoma-detection-mt.s3.nl-ams.scw.cloud/isic_2019_tfrecords_384.zip](https://melanoma-detection-mt.s3.nl-ams.scw.cloud/isic_2019_tfrecords_384.zip)
- [https://melanoma-detection-mt.s3.nl-ams.scw.cloud/melanoma_tfrecords_384x384.zip](https://melanoma-detection-mt.s3.nl-ams.scw.cloud/melanoma_tfrecords_384x384.zip)
