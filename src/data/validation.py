import numpy as np

from data.verify_tf_records import display_batch_of_images
from data.load_tf_records import get_training_dataset, augmentation_pipeline


def validate_data(filenames, batch_size,):
    example_dataset = get_training_dataset(
        filenames, batch_size, augment=False)
    example_dataset = example_dataset.unbatch().batch(20)
    example_batch = iter(example_dataset)
    image_batch, label_batch = next(example_batch)

    print("show some images from the dataset")
    print(display_batch_of_images((image_batch, label_batch)))

    print("show some augmented images from the dataset")
    image_batch, label_batch = augmentation_pipeline(image_batch, label_batch)
    display_batch_of_images((image_batch, label_batch))

    # images are in uint8 format with values between 5 and 232
    # image should be scaled between -1 and 1 => needed for resnetV2
    first_image = image_batch[0]
    print("Image min & max values", np.min(first_image), np.max(first_image))
    print("Image dtype", first_image.dtype)


def get_class_distribution_of_dataset(df):
    total_img = df['target'].size
    malignant_cases = np.count_nonzero(df['target'])
    benign_cases = total_img - malignant_cases

    print("Samples in dataset", total_img)
    print("Total cases of class 1", malignant_cases)
    print("Positive cases in dataset", 100 * malignant_cases / total_img)

    initial_bias = np.log([malignant_cases/benign_cases])
    print("Bias", initial_bias)

    weight_for_0 = (1 / benign_cases)*(total_img)/2.0
    weight_for_1 = (1 / malignant_cases)*(total_img)/2.0

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return malignant_cases, benign_cases
