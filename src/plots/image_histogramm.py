import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import numpy


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
    """Returns the pixel distribution of an image

    Args:
        image_path (string)

    Returns:
        List: Mean and Std
    """
    raw_image = plt.imread(image_path)
    mean = raw_image.mean()
    std = raw_image.std()

    return [mean, std]


def get_means_and_stds(df):
    """Returns means and stds of images in a dataframe

    Args:
        df (pandas dataframe): Must have 'image_path' column

    Returns:
        Tuple: Tuple with lists of means and stds of images
    """
    means = []
    stds = []

    for i in tqdm(range(df.shape[0])):
        image_path = df.iloc[i].image_path
        mean, std = get_pixel_distribution(image_path)
        means.append(mean)
        stds.append(std)

    return means, stds


def image_hist(image_path, image_title, mean=None):
    """ Helper function to plot the pixel intensitiy distribution for rgb images
        Parameters:
            image_path (str) The path to the image
            image_title (str) The title of the plot
        Returns:
            Null
    """
    f = plt.figure(figsize=(16, 8))
    f.add_subplot(1, 2, 1)

    raw_image = plt.imread(image_path)
    plt.imshow(raw_image, cmap='gray')
    plt.colorbar()
    plt.title(image_title)
    print(f"Image dimensions:  {raw_image.shape[0],raw_image.shape[1]}")
    print(
        f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}")
    print(
        f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}")

    f.add_subplot(1, 2, 2)

    _ = plt.hist(raw_image[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
    _ = plt.hist(raw_image[:, :, 1].ravel(),
                 bins=256, color='Green', alpha=0.5)
    _ = plt.hist(raw_image[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])

    if mean:
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)

    plt.show()
