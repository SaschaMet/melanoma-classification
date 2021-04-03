import logging
import requests
from PIL import Image
import tensorflow as tf
import efficientnet.keras as efn


def get_model():
    model = tf.keras.models.load_model("final_model.h5")
    model.load_weights("final_weights.hdf5")
    return model


def predict(image_url):
    try:
        # 1. get the image form the url
        response = requests.get(image_url, stream=True).raw
        image = Image.open(response)
        # 2. resize the image / format the image
        image = image.resize((512, 512))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)

        # 3. predict
        model = get_model()
        prediction = model.predict(image, batch_size=1)
        prediction = prediction[0][0]

        # 4. return the prediction
        return prediction
    except Exception as e:
        logging.error('Error at %s', 'division', exc_info=e)
        return "error"
