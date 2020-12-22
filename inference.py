from os import replace
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.applications.resnet50 import preprocess_input


def load_model(model_path, weight_path):
    with open(model_path, 'r') as json_file:
        model_file = json_file.read()
        model = keras.models.model_from_json(model_file)
        model.load_weights(weight_path)

    return model


def predict_image(model, img):
    pred = model.predict(img, verbose=1, steps=3)
    return pred
    #print("pred", pred)
    # if pred > thresh:
    # return 'malignant', pred
    # else:
    # return 'non malignant', pred


def start_prediction(timestamp):

    model_path = "./" + timestamp + "-model.json"
    weight_path = "./" + timestamp + "-model.hdf5"

    thresh = 0.51
    IMG_SIZE = (224, 224)
    my_model = load_model(model_path, weight_path)

    test_images = [
        "../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0052060.jpg",
        "../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0052349.jpg"
    ]

    images = []

    for img in test_images:

        img_name = img.replace("/input", "")

        img = keras.preprocessing.image.load_img(
            path=img, target_size=IMG_SIZE)

        if img is None:
            print("No image found for", img_name)
            continue

        img_arr = keras.preprocessing.image.img_to_array(img)
        img_arr = img_arr.astype('float32')
        img_arr = img_arr / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr, "channels_last")

        # plt.imshow(img, cmap='gray')
        # plt.axis("off")
        # plt.show()

        pred = predict_image(my_model, img_arr)
        print(pred)
