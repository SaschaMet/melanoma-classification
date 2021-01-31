import matplotlib.pyplot as plt


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
