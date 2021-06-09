# Model Service

This guide will provide an overview of how to deploy, create and adapt the service.

- How to deploy?
There is already a Docker image available on dockerhub.com. Using this image, you can deploy the service with two commands:

1. Download the docker image: `docker pull smet/melanoma-service`
2. Start the docker container on port 80: `docker run -d --name melanoma-service -p 80:80 smet/ melanoma-service`

Another method to start the container is to use the provided `run_docker_image.sh` script.

Docker has many more options on how to start a container. Detailed documentation is available here: [https://docs.docker.com/](https://docs.docker.com/)

- How to create your own service?
The service consists of three python files:

1. `main.py`
This file will create a webserver and provide the REST-API used to communicate with the models. The service itself provides two routes `/predict` and `/cam`. Each route will call one of the other two python files.

2. `model.py`
This file is executed when the `/predict` route is called. This script will download the provided image, it will get a prediction from the model(s), and it will return the prediction.

3. `grad_cam.py`
This file is executed when the `/cam` route is called. It will produce and return a Grad-Cam plot from the provided image.

- How to adapt the service to my own needs?
You can adapt the service by adjusting the mentioned files. By executing the `build_docker_image.sh` script, a docker image will be created for you. With the `run_docker_image.sh`, you can run the image. By executing the `push_docker_image.sh` script, you can push the image to dockerhub.com (Note: you will have to change the name of your image to do this).
