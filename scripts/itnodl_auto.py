#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
TASK 2 - AUTOENCODER

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os
from math import ceil

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt

import itnodl_data as dat
import itnodl_help as hlp
from itnodl_data import squash, lift
from itnodl_help import log, set_up_model_directory


#################
# Architectures #
#################

def linear_auto_architecture(image_size: int, optimizer: str, loss: str, encoding_dim: int) -> Model():
    """
    Build architecture for linear autoencoder with single layer.

    :param image_size: data size (image dim x image dim x color channels)
    :param optimizer: optimizer function
    :param loss: loss function
    :return:
    """

    # Model parameters
    input_shape = (image_size,)

    # Build model
    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, input_shape=input_shape, activation='linear',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(Dense(image_size, activation='linear',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))

    # Compile model
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder


def convolutional_auto_architecture(image_dim: int, optimizer: str, loss: str, encoding_dim: int) -> Model():
    """
    Build architecture for deep convolutional autoencoder.

    :param image_dim: dimension of image (total size = image dim x image dim x color channels)
    :param optimizer: optimizer function
    :param loss: loss function
    :return:
    """

    # Model parameters
    input_shape = (image_dim, image_dim, 3)

    # Build model
    autoencoder = Sequential()
    autoencoder.add(Conv2D(2 * encoding_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))

    autoencoder.add(Conv2D(encoding_dim, (3, 3), padding='same', activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

    autoencoder.add(Conv2D(encoding_dim, (3, 3), padding='same', activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(UpSampling2D((2, 2)))

    autoencoder.add(Conv2D(2 * encoding_dim, (3, 3), padding='same', activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(UpSampling2D((2, 2)))

    autoencoder.add(Conv2D(3, (10, 10), padding='same', activation='sigmoid',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())

    # Compile model
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder


###################
# Build and train #
###################

def build_autoencoder(model_name: str, convolutional: bool, train: bool,
                      x_tr: np.ndarray, x_va: np.ndarray,
                      compression_factor=32, epochs=100,
                      optimizer="adam", loss="mean_squared_error",
                      patience=5) -> \
        (Sequential, Sequential, float):
    """
    Build an autoencoder.

    :param model_name: name used in file creation
    :param x_tr: training images
    :param x_va: validation images
    :param encoding_dim: number of nodes in encoder layer (i.e. the bottleneck)
    :param epochs: number of epochs to train for
    :param optimizer: optimizer to use in training
    :param loss: loss function to use in training
    :param patience: number of epochs without improvement before early stop
    :return: autoencoder model, encoder model, decoder model, compression factor
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
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))

    # Build model path
    model_path = os.path.join(os.pardir, "models", "autoencoders", full_model_name + ".h5")
    architecture_path = os.path.join(os.pardir, "models", "autoencoders", "architecture",
                                     full_model_name + "_architecture.png")

    # Keep track of history
    history = []
    history_path = os.path.join(os.pardir, "models", "autoencoders", "history", full_model_name + "_history.npy")

    # Try loading the model, ...
    try:

        autoencoder = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path).tolist()
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except:

        if convolutional:
            log("Training deep convolutional autoencoder.", lvl=2)
            autoencoder = convolutional_auto_architecture(image_dim=image_dim, optimizer=optimizer, loss=loss,
                                                          encoding_dim=encoding_dim)
        else:
            log("Training linear autoencoder.", lvl=2)
            autoencoder = linear_auto_architecture(image_size=image_size, optimizer=optimizer, loss=loss,
                                                   encoding_dim=encoding_dim)

        # Print to log
        log("-- image dimension: {}, image size: {}, compression factor {}".
            format(image_dim, image_size, compression_factor), lvl=3)

    # Train the model (either continue training the old model, or train the new one)
    if train:

        # Flatten image data for linear model
        if not convolutional:
            # Flatten
            x_tr = squash(x_tr)
            x_va = squash(x_va)

        # Training parameters
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
        mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        tb = TensorBoard(log_dir='/tmp/' + model_name + '_im' + str(image_dim) + 'comp' + str(int(compression_factor)))

        # Train model
        history.append(
            autoencoder.fit(x_tr, x_tr,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose,
                            validation_data=(x_va, x_va),
                            callbacks=[es, mc, tb])
        )

        # Save model and history
        # autoencoder.save(autoencoder_path) # <- already stored at checkpoint
        np.save(file=history_path, arr=history)

    # Visual aid
    plot_model(autoencoder, to_file=architecture_path, show_layer_names=True, show_shapes=True)

    # Get intermediate output at encoded layer
    if convolutional:
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='encoder').output)
    else:
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=0).output)

    # See effect of encoded representations on output (eigenfaces?)
    '''decoder = Model(inputs=Input(shape=(encoding_dim,)), outputs=autoencoder.output)
    plot_model(autoencoder, to_file=os.path.join(os.pardir, "models", "decoder.png"), show_layer_names=True, show_shapes=True)'''

    # Size of encoded representation
    log("Compression factor is {}, encoded vector length is {}.".format(compression_factor, encoding_dim), lvl=3)

    return autoencoder, encoder, history


#############
# Visualize #
#############

def plot_autoencoder_results(autoencoder: Sequential, convolutional: bool, model_name: str,
                         compression_factor: float, x: np.ndarray, examples=5, save=True, plot=True):
    """
    Plot a number of examples: compare original and reconstructed images.

    :param autoencoder: the model to evaluate
    :param compression_factor: the model's compression factor
    :param x: the data to pass through
    :param examples: number of images to visualize
    :return:
    """

    # Whip input into training shape
    if not convolutional:
        x = squash(x)

    # Check results
    x_pred = autoencoder.predict(x=x)

    # Whip input and output into image shape
    x = np.clip(lift(x), 0, 1)
    x_pred = np.clip(lift(x_pred), 0, 1)

    # Get image dimension
    image_dim = x_pred.shape[1]

    # Plot parameters
    plot_count = examples * 2
    row_count = 2
    col_count = int(ceil(plot_count / row_count))

    # Initialize axes
    fig, subplot_axes = plt.subplots(row_count,
                                     col_count,
                                     squeeze=False,
                                     sharex='all',
                                     sharey='all',
                                     figsize=(12, 6))
    # Fill axes
    for i in range(plot_count):

        row = i // col_count
        col = i % col_count

        original_image = x[col]
        reconstructed_image = x_pred[col].reshape(image_dim, image_dim, 3) if not convolutional else x_pred[col]

        ax = subplot_axes[row][col]
        if row == 0:
            ax.set_title("Original")
            ax.imshow(original_image)
        else:
            ax.set_title("Reconstructed")
            ax.imshow(reconstructed_image)

        ax.axis('off')

    # General make-up
    sup_string = ('Deep CNN' if convolutional else 'Linear') + \
                 ' autoencoder - image dim {}, compression factor {}'.format(image_dim, compression_factor)
    plt.suptitle(sup_string,
                 fontweight='bold', fontsize='12')
    plt.tight_layout()

    # Save Figure
    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))

    # Build model path
    model_path = os.path.join(os.pardir, "models", "autoencoders", "plots", full_model_name + ".png")

    # Show 'n tell
    if save: plt.savefig(model_path)
    if plot: plt.show()


########
# MAIN #
########

if __name__ == "__main__":
    """Main function."""

    # Let's go
    log("AUTOENCODERS", title=True)

    # Set parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('autoencoders')

    # Model approaches
    linear, linear_train = True, True
    convolutional, convolutional_train = True, True

    # Loop over these dimensions
    image_dims = ()
    compression_factors = (16, 32, 48,)

    dims = [(i, c) for i in image_dims for c in compression_factors]

    for image_dim, compression_factor in dims:

        # Calculate extra parameters
        image_size = image_dim ** 2 * 3

        # Get the data
        (x_tr, y_tr), (x_va, y_va) = dat.pipeline(image_dim=image_dim)

        # Build linear autoencoder
        if linear:
            # Log
            log("Linear autoencoder for {id}x{id} images, compression {c}".format(id=image_dim,
                                                                                  c=compression_factor), title=True)
            # Build
            lin_auto, lin_enc, lin_hist = build_autoencoder(model_name='lin_auto',
                                                            convolutional=False,
                                                            train=linear_train,
                                                            x_tr=x_tr, x_va=x_va,
                                                            compression_factor=compression_factor,
                                                            epochs=150,
                                                            patience=15)

            # Visualize
            plot_autoencoder_results(model_name='lin_auto', autoencoder=lin_auto, convolutional=False,
                                 compression_factor=compression_factor,
                                 x=x_va, save=True, plot=False)

        # Build deep convolutional auto encoder
        if convolutional:
            # Log
            log("Deep convolutional autoencoder for {id}x{id} images, compression {c}".format(id=image_dim,
                                                                                              c=compression_factor),
                title=True)

            # Build
            dcnn_auto, dcnn_enc, dcnn_hist = build_autoencoder(model_name='conv_auto',
                                                               convolutional=True,
                                                               train=convolutional_train,
                                                               x_tr=x_tr, x_va=x_va,
                                                               compression_factor=compression_factor,
                                                               epochs=100,
                                                               patience=10)

            # Train
            plot_autoencoder_results(model_name='conv_auto', autoencoder=dcnn_auto, convolutional=True,
                                 compression_factor=compression_factor,
                                 x=x_va, save=True, plot=False)
