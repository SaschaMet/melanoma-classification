from model.clr_callback import CyclicLR
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def get_model_callbacks(steps_per_epoch, base_lr, max_lr, verbose_level, save_output, timestamp, use_tensorboard=False, use_clr=False):
    # model callbacks
    callback_list = []

    if use_clr:
        print("use cyclical learning rate callback")
        # initialize the cyclical learning rate callback
        clr = CyclicLR(
            mode="triangular",
            base_lr=base_lr,
            max_lr=max_lr,
            step_size=steps_per_epoch/8
        )
        callback_list.append(clr)

    # if the model does not improve for 10 epochs, stop the training
    stop_early = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True
    )
    callback_list.append(stop_early)

    # add tensorboard when we can use it
    if use_tensorboard:
        log_dir = "logs/fit/" + timestamp
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callback_list.append(tensorboard_callback)

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
            mode='min',
        )
        # append the checkpoint callback to the callback list
        callback_list.append(checkpoint)

    return callback_list
