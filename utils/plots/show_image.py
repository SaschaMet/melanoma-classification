import matplotlib.pyplot as plt

FIGSIZE = (10, 10)


def show_single_image(image_path, image_array=None, show_axis="off"):
    """Displays a single image. Either image path or image array must be given

    Args:
        image_path (str): Path to the image
        image_array (str): Image Array
        show_axis (str, optional): Display the axis? Defaults to "off".
    """
    img = image_array
    if image_path:
        img = plt.imread(image_path)
    plt.figure(figsize=FIGSIZE)
    plt.imshow(img, cmap='gray')
    plt.axis(show_axis)


def show_images(img_paths, rows, columns):
    """Shows multiple images in a subplot

    Args:
        img_paths (List of strings): List of image paths
        rows (number): Number rows
        columns (number): Number of columns
    """
    total_images = rows * columns
    plt.figure(figsize=FIGSIZE)
    for i in range(total_images):
        plt.subplot(3, 3, i + 1)
        img_path = img_paths[i]
        img = plt.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
