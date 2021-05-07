from model import predict
from grad_cam import get_grad_cam
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def read_root(body: dict):
    prediction = predict(
        body['image_url'],
        body['number_of_models']
    )
    return dict([("prediction", str(prediction))])


@app.post("/cam")
async def read_root(body: dict):
    get_grad_cam(body['image_url'])
    return FileResponse("./grad_cam.png")
