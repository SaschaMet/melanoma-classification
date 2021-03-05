from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn


def get_model_callbacks(verbose_level, save_output, timestamp):
    # model callbacks
    callback_list = []

    # if the model does not improve for 10 epochs, stop the training
    stop_early = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=10,
        restore_best_weights=True
    )
    callback_list.append(stop_early)

    # add tensorboard
    log_dir = "logs/fit/" + timestamp
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_list.append(tensorboard_callback)

    # learning rate decay
    exponential_decay_fn = exponential_decay(0.01, 20)
    lr_scheduler = LearningRateScheduler(exponential_decay_fn)
    callback_list.append(lr_scheduler)

    # if the output of the model should be saved, create a checkpoint callback function
    if save_output:
        # set the weight path for saving the model
        weight_path = "./" + timestamp + "-model.hdf5"
        # create the model checkpoint callback to save the model weights to a file
        checkpoint = ModelCheckpoint(
            weight_path,
            save_weights_only=True,
            verbose=verbose_level,
            save_best_only=True,
            monitor='val_loss',
            overwrite=True,
            mode='auto',
        )
        # append the checkpoint callback to the callback list
        callback_list.append(checkpoint)

    return callback_list
