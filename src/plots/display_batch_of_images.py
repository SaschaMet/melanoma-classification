import math
import matplotlib.pyplot as plt


def batch_to_numpy_images_and_labels(data):
    """Returns a list of images and labels from a data batch

    Args:
        data (batch): A data batch with labels and images

    Returns:
        tuple: A tuple of lists of labels and images
    """
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label):
    """Returns "OK" ofr "NO" for a label and a prediction

    Args:
        label (string)
        correct_label (string)

    Returns:
        string
    """
    CLASSES = [0, 1]
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct


def display_one_image(image, title, subplot, red=False, titlesize=16):
    """Displays a single image in subplot

    Args:
        image
        title
        subplot

    Returns:
        subplot
    """
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2),
                  color='red' if red else 'black', fontdict={'verticalalignment': 'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images(databatch, predictions=None, unbatched=False):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(
        databatch) if not unbatched else databatch
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows

    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = 'Benign' if label == 0 else 'Malignant'
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        # magic formula tested to work from 1x1 to 10x10 images
        dynamic_titlesize = FIGSIZE*SPACING/max(rows, cols)*40+3
        subplot = display_one_image(
            image, title, subplot, not correct, titlesize=dynamic_titlesize)

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
