# get each image in the train set
# get the distribution of rgb pixel
# calculate the mean and standard deviation
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


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
