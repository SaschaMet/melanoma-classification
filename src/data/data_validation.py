import numpy as np

from data.verify_tf_records import display_batch_of_images
from data.load_tf_records import get_training_dataset, augment_image


def get_pixel_distribution(image_batch):
    # images are in float32 format with normalized values
    for i in range(10):
        image = image_batch[i]
        print("min:", np.min(image), " -  max:", np.max(image))

    print(image.dtype)


def validate_data(batch_size=15):
    example_dataset = get_training_dataset(augment=False)
    example_dataset = example_dataset.unbatch().batch(batch_size)

    example_batch = iter(example_dataset)
    image_batch, label_batch = next(example_batch)
    print("show some images from the dataset")
    display_batch_of_images((image_batch, label_batch))

    print("show some augmented images from the dataset")
    augmented_images = [augment_image(x, augment=True) for x in image_batch]
    augmented_images = [np.clip(x, 0, 1) for x in augmented_images]
    labels = [l.numpy() for l in label_batch]
    display_batch_of_images((augmented_images, labels), unbatched=True)

    # images are in uint8 format with values between 5 and 232
    # image should be scaled between -1 and 1 => needed for resnetV2
    first_image = image_batch[0]
    print("Image min & max values", np.min(first_image), np.max(first_image))
    print("Image dtype", first_image.dtype)


def get_class_distribution_of_dataset(data_batch, number_of_images, iterations=50):
    # Iterate over n batches to get the class distribution
    benign_cases = 0
    malignant_cases = 0

    for _ in range(0, iterations):
        _, y = next(data_batch)
        for label in y.numpy():
            if int(label) == 0:
                benign_cases = benign_cases + 1
            else:
                malignant_cases = malignant_cases + 1

    initial_bias = np.log([malignant_cases/benign_cases])

    weight_for_0 = (1 / benign_cases)*(number_of_images)/2.0
    weight_for_1 = (1 / malignant_cases)*(number_of_images)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print("benign_cases", benign_cases)
    print("malignant_cases", malignant_cases)
    print("ratio", round(malignant_cases / benign_cases, 2))
    print("initial_bias", initial_bias)

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return initial_bias, class_weight
