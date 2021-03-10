import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler


def get_lr_callback(strategy, epochs):
    # Source: https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/07_Keras_Flowers_TPU_xception_fine_tuned_best.ipynb#scrollTo=M-ID7vP5mIKs
    start_lr = 1e-5
    min_lr = 1e-6
    max_lr = 1e-4
    rampup_epochs = 6
    sustain_epochs = 1
    exp_decay = 0.8

    def lrfn(epoch):
        def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
            if epoch < rampup_epochs:
                lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
            elif epoch < rampup_epochs + sustain_epochs:
                lr = max_lr
            else:
                lr = (max_lr - min_lr) * exp_decay**(epoch -
                                                     rampup_epochs-sustain_epochs) + min_lr
            return lr

        return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)

    lr_callback = LearningRateScheduler(
        lambda epoch: lrfn(epoch), verbose=True)

    print("learning rate decay")
    rng = [i for i in range(epochs)]
    y = [lrfn(x) for x in rng]
    print(plt.plot(rng, [lrfn(x) for x in rng]))

    return lr_callback


def get_model_callbacks(strategy, epochs, verbose_level, save_output, timestamp, use_tensorboard=False):
    # model callbacks
    callback_list = []

    lr_callback = get_lr_callback(strategy, epochs)
    callback_list.append(lr_callback)

    # if the model does not improve for 10 epochs, stop the training
    stop_early = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,
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
            monitor='val_auc',
            overwrite=True,
            mode='max',
        )
        # append the checkpoint callback to the callback list
        callback_list.append(checkpoint)

    return callback_list
