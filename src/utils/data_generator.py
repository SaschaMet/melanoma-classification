from keras.preprocessing.image import ImageDataGenerator


def get_training_gen(df, seed, img_size, batch_size):
    """ Factory function to create a training image data generator

    Parameters:
        df (dataframe): Training dataframe

    Returns:
        Image Data Generator function
    """
    # prepare images for training
    train_idg = ImageDataGenerator(
        rescale=1 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.15,
        width_shift_range=0.15,
        shear_range=0.15,
        rotation_range=90,
        zoom_range=0.20,
        fill_mode='nearest'
    )

    train_gen = train_idg.flow_from_dataframe(
        seed=seed,
        dataframe=df,
        directory=None,
        x_col='image_path',
        y_col=['target_0', 'target_1'],
        class_mode='raw',
        shuffle=True,
        target_size=img_size,
        batch_size=batch_size,
        validate_filenames=False
    )

    return train_gen


def get_validation_gen(df, seed, img_size, batch_size):
    """ Factory function to create a validation image data generator

    Parameters:
        df (dataframe): Validation dataframe

    Returns:
        Image Data Generator function
    """
    # prepare images for validation
    val_idg = ImageDataGenerator(rescale=1. / 255.0)
    val_gen = val_idg.flow_from_dataframe(
        seed=seed,
        dataframe=df,
        directory=None,
        x_col='image_path',
        y_col=['target_0', 'target_1'],
        class_mode='raw',
        shuffle=False,
        target_size=img_size,
        batch_size=batch_size,
        validate_filenames=False
    )

    return val_gen
