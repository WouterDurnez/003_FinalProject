#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
TASK 4 - SEGMENTATION

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import itnodl_data as dat
import itnodl_help as hlp
from itnodl_data import show_some
from itnodl_help import log, set_up_model_directory


#############
# Dice loss #
#############

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    # some implementations don't square y_pred
    denominator = tf.reduce_sum(y_true + tf.square(y_pred))

    return np.abs(numerator / (denominator + tf.keras.backend.epsilon()))


#################
# Architectures #
#################

def convolutional_segm_architecture(image_dim: int, optimizer: str, loss: str, encoding_dim: int, mono=True) -> Model():
    """
    Build architecture for deep convolutional autoencoder.

    :param image_dim: dimension of image (total size = image dim x image dim x color channels)
    :param optimizer: optimizer function
    :param loss: loss function
    :return: segmentation network architecture
    """

    # Model parameters
    input_shape = (image_dim, image_dim, 3)

    # Build model
    segnet = Sequential()
    segnet.add(Conv2D(2 * encoding_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                      kernel_initializer='random_uniform', bias_initializer='zeros'))
    segnet.add(BatchNormalization())
    segnet.add(MaxPooling2D((2, 2), padding='same'))

    segnet.add(Conv2D(encoding_dim, (3, 3), padding='same', activation='relu',
                      kernel_initializer='random_uniform', bias_initializer='zeros'))
    segnet.add(BatchNormalization())
    segnet.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

    segnet.add(Conv2D(encoding_dim, (3, 3), padding='same', activation='relu',
                      kernel_initializer='random_uniform', bias_initializer='zeros'))
    segnet.add(BatchNormalization())
    segnet.add(UpSampling2D((2, 2)))

    segnet.add(Conv2D(2 * encoding_dim, (3, 3), padding='same', activation='relu',
                      kernel_initializer='random_uniform', bias_initializer='zeros'))
    segnet.add(BatchNormalization())
    segnet.add(UpSampling2D((2, 2)))

    if mono:
        segnet.add(Conv2D(1, (10, 10), padding='same', activation='sigmoid',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
    else:
        segnet.add(Conv2D(3, (10, 10), padding='same', activation='sigmoid',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))

    segnet.add(BatchNormalization())

    # Compile model
    segnet.compile(optimizer=optimizer, loss=loss)

    segnet.summary()

    return segnet


###################
# Build and train #
###################

def build_segm_net(model_name: str, train: bool,
                   x_tr: np.ndarray, y_tr: np.ndarray,
                   x_va: np.ndarray, y_va: np.ndarray,
                   mono=True,
                   compression_factor=32, epochs=100,
                   optimizer="adam", loss='mean_squared_error',
                   patience=5) -> (Sequential, History):
    """
    Build and train segmentation net

    :param model_name: name to be used in file output
    :param train: (continue and) train the model?
    :param x_tr: training images
    :param y_tr: training targets
    :param x_va: validation images
    :param y_va: validation targets
    :param compression_factor: (sic)
    :param epochs: training duration
    :param optimizer: (sic)
    :param loss: (sic)
    :param patience: epochs to wait without seeing validation improvement
    :return: model and its history
    """
    # Model parameters
    image_dim = x_tr.shape[1]
    image_size = image_dim ** 2 * 3
    encoding_dim = int(image_size / compression_factor)

    # Set parameters
    batch_size = 32
    if hlp.LOG_LEVEL == 3:
        verbose = 1
    elif hlp.LOG_LEVEL == 2:
        verbose = 2
    else:
        verbose = 0

    # Full model name for file output
    full_model_name = "{}_im_dim{}-comp{}_{}_mono{}".format(model_name, image_dim, compression_factor, loss, mono)

    # Build model path
    model_path = os.path.join(os.pardir, "models", "segmentation", full_model_name + ".h5")
    architecture_path = os.path.join(os.pardir, "models", "segmentation", "architecture",
                                     full_model_name + "_architecture.png")

    # Keep track of history
    history = []
    history_path = os.path.join(os.pardir, "models", "segmentation", "history", full_model_name + "_history.npy")

    # Try loading the model, ...
    try:

        segnet = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path).tolist()
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except:

        log("Building segmentation network.")
        segnet = convolutional_segm_architecture(image_dim=image_dim, optimizer=optimizer, loss=loss,
                                                 encoding_dim=encoding_dim)

        # Print model info
        log("Network parameters: image dimension {}, image size {}, compression factor {}.".
            format(image_dim, image_size, compression_factor), lvl=3)

    # Train the model (either continue training the old model, or train the new one)
    if train:
        log("Training deep convolutional segmentation network.", lvl=2)

        # Callbacks
        if patience == 0:
            patience = epochs
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
        mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        tb = TensorBoard(log_dir='/tmp/' + model_name + '_im' + str(image_dim) +
                                 'comp' + str(int(compression_factor)) + 'mono' + str(mono))

        # Data augmentation to get the most out of our images
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(x_tr)

        # Train model using data augmentation
        history = segnet.fit_generator(datagen.flow(x_tr, y_tr, batch_size=32),
                                       epochs=epochs,
                                       steps_per_epoch=x_tr.shape[0] // 32,
                                       verbose=verbose,
                                       validation_data=(x_va, y_va),
                                       callbacks=[es, mc, tb])

        # Save model and history
        # autoencoder.save(autoencoder_path) # <- already stored at checkpoint
        np.save(file=history_path, arr=history)

    # Visual aid
    plot_model(segnet, to_file=architecture_path, show_layer_names=True, show_shapes=True)

    # Size of encoded representation
    log("Compression factor is {}, encoded vector length is {}.".format(compression_factor, encoding_dim), lvl=3)

    return segnet, history


########
# MAIN #
########

if __name__ == "__main__":

    # Let's go
    log("SEGMENTATION", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('segmentation')

    # Dashboard
    image_dim = 48
    compression_factor = 48
    train = True
    epochs = 500
    patience = 50
    loss = 'mean'

    # Set remaining parameters
    segmentation_name = 'conv_segm'
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    # Get the data
    _, data = dat.pipeline(image_dim=image_dim, class_data=True, segm_data=True, mono=True)

    # USING TRAINVAL TO TRAIN BECAUSE THERE ARE MORE PICTURES!
    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']

    # Train segmentation network
    segnet, history = build_segm_net('conv_segm', train=train, x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va,
                                     compression_factor=compression_factor, epochs=epochs,
                                     optimizer='adam', loss=loss, patience=patience, mono=True)

    x_te, y_te, y_te_src = data['x_test'], data['y_test'], data['y_test_source']

    y_te_pred = segnet.predict(x=x_te)
    y_tr_pred = segnet.predict(x=x_tr)

    show_some(image_list=[x_te, y_te_src, y_te, y_te_pred],
              subtitles=["Original images", "Target segmentation", "Target segmentation (threshold)",
                         "Predicted segmentation"],
              n=5, title="Segmentation prediction", random=False,
              save=True, save_name="conv_segm_result_im{}comp{}".format(image_dim, compression_factor))

    show_some(image_list=[x_tr, y_tr, y_tr_pred],
              subtitles=["Original images", "Target segmentation",
                         "Predicted segmentation"],
              n=5, title="Segmentation prediction", random=False,
              save=True, save_name="conv_segm_result_im{}comp{}".format(image_dim, compression_factor))
