import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import efficientnet.keras as efn


def get_model_21():
    model = tf.keras.models.load_model("./models/b5_Seed_21.h5")
    return model


def get_grad_cam(image_url):

    # get the image form the url
    response = requests.get(image_url, stream=True).raw
    image = Image.open(response)

    # resize the image / format the image
    image = image.resize((384, 384))
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Get the CNNs output layer
    model = get_model_21()
    efficientnet_model = False
    for layer in model.layers:
        if layer.name == "efficientnet-b5":
            efficientnet_model = layer

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [efficientnet_model.inputs], [efficientnet_model.get_layer(
            'top_conv').output, efficientnet_model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(
            np.expand_dims(image, axis=0))
        class_channel = preds[:, round(np.mean(tf.argmax(preds[0]).numpy()))]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((384, 384))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(superimposed_img)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image / 255)

    plt.savefig('grad_cam.png')
    return True
