docker stop melanoma-service
docker rm melanoma-service
docker run -d --name melanoma-service -p 80:80 smet/melanoma-service