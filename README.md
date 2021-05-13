# Melanoma diagnosis service

## Identify melanoma in lesion images

This repository holds the source code for my master thesis
"Model as a Service: Development of a prototype for computer-aided skin cancer diagnosis".

To identify a melanoma, several neural networks were trained based on the Kaggle dataset from the "SIIM-ISIC Melanoma Classification" competition: <https://www.kaggle.com/c/siim-isic-melanoma-classification/overview>

The final result is an ensemble from two EfficientNet B5 models. It achieved an ROC/AUC score of 0.9514.

The training process as well as the jupyter notebooks for all models can be found here: <https://www.comet.ml/saschamet/master-thesis/view/oReT9Mkucm9UUH1LMinGOQmyl>

A Kaggle notebook showing the training process of a single EfficientNet B5 model is available here: <https://www.kaggle.com/saschamet/melanoma-efficientnetb5-noisy-student>

## Model as a Service

The ensemble can be deployed with a Docker image. The image can be retrieved here: <https://hub.docker.com/r/smet/melanoma-service>

The source code can be found in the `/service` directory.

To start the service, execute the following two commands: <br>
`docker pull smet/melanoma-service`
<br>
`docker run -d --name melanoma-service -p 80:80 smet/melanoma-service`
<br>
The service is now available on port 80.
<br><br>
There are two routes. The first route returns simply a prediction. The second route returns a grad cam image.

1. /predict
<br>
Method: POST;
<br>
Parameters:

- image_url: URL of an image to predict
- number_of_models: Either 1 or 2 - How many models should be used for the prediction.
<br><br>

2. POST ​/cam
<br>
Method: POST
<br>
Parameters:

- image_url: URL of an image to predict

## Important Note

This application is created for scientific purposes only!
