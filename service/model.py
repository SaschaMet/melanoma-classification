import logging
import requests
from PIL import Image
import tensorflow as tf
import efficientnet.keras as efn


def get_model_21():
    """Helper function to load a model

    Returns:
        tf model: Tensorflow model
    """
    model = tf.keras.models.load_model("./models/b5_Seed_21.h5")
    return model


def get_model_42():
    """Helper function to load a model

    Returns:
        tf model: Tensorflow model
    """
    model = tf.keras.models.load_model("./models/b5_Seed_42.h5")
    return model


def predict(image_url, number_of_models=1):
    """Get model prediction

    Args:
        image_url (string): A URL to an image to get a prediction from
        number_of_models (int, optional): Number of models to use for the prediction. Can be 1 or 2. Defaults to 1

    Returns:
        int: Prediction probability of beeing a malignant melanoma
    """
    try:
        # 1. get the image form the url
        response = requests.get(image_url, stream=True).raw
        image = Image.open(response)

        # 2. resize the image / format the image
        image = image.resize((384, 384))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, axis=0)

        # 3. predict
        model = get_model_21()
        prediction = model.predict(image, batch_size=1)
        prediction = prediction[0][0]
        logging.info("prediction", prediction)
        print("prediction", prediction)

        if number_of_models == 2:
            model = get_model_42()
            prediction_2 = model.predict(image, batch_size=1)
            prediction_2 = prediction_2[0][0]
            print("prediction_2", prediction_2)
            logging.info("prediction_2", prediction_2)

            prediction = (prediction + prediction_2) / 2

        prediction = round(prediction, 4) * 100

        # 4. return the prediction
        return prediction
    except Exception as e:
        logging.error('Error at %s', exc_info=e)
        return "error"
