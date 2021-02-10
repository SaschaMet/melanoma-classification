# get each image in the train set
# get the distribution of rgb pixel
# calculate the mean and standard deviation
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import numpy

# Source: https://stackoverflow.com/questions/9506841/using-python-pil-to-turn-a-rgb-image-into-a-pure-black-and-white-image


def binarize_image(img_path, threshold=200):
    """Binarize an image

    Args:
        img_path (string): Path to the image
        threshold (int, optional): Threshold for binary conversion. Defaults to 200.
    """
    image_file = Image.open(img_path)
    image = image_file.convert('L')
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    return image


def binarize_array(numpy_array, threshold):
    """Binarize a numpy array

    Args:
        numpy_array (array): Array of pixels
        threshold (int): Threshold for binary conversion

    Returns:
        [array]: binarized numpy array
    """
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def get_pixel_distribution(image_path):
    raw_image = plt.imread(image_path)

    mean = raw_image.mean()
    std = raw_image.std()

    return [mean, std]


def get_means_and_stds(df):
    means = []
    stds = []

    for i in tqdm(range(df.shape[0])):
        image_path = df.iloc[i].image_path
        mean, std = get_pixel_distribution(image_path)
        means.append(mean)
        stds.append(std)

    return means, stds
