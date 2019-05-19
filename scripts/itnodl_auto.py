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

import itnodl_data as dat
import itnodl_help as hlp
import numpy as np
import tensorflow as tf
from itnodl_data import squash, lift
from itnodl_help import log, make_folders
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt


def prep_input(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing of input ndarray.

    :param x: array of indices (shape: n x image_size x image_size x 3)
    :return: mean corrected x
    """
    x_new = x[:, ] - mean
    return x_new


def linear_autoencoder(x_tr: np.ndarray, x_va: np.ndarray,
                       model_name='linear_autoencoder',
                       encoding_dim=32, epochs=100,
                       optimizer="adam", loss="mean_squared_error") -> \
        (Sequential, Sequential, float):
    """
    Build a linear autoencoder.

    :param model_name: name used in file creation
    :param x_tr: training images
    :param x_va: validation images
    :param encoding_dim: number of nodes in encoder layer (i.e. the bottleneck)
    :param epochs: number of epochs to train for
    :param optimizer: optimizer to use in training
    :param loss: loss function to use in training
    :return: autoencoder model, encoder model, decoder model, compression factor
    """

    # Get image dimension
    image_dim = x_tr.shape[1]
    image_size = image_dim ** 2 * 3

    # Set parameters
    batch_size = 32

    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-en_dim' + str(encoding_dim)

    # Build model path
    model_path = os.path.join(os.pardir, "models", full_model_name + ".h5")
    best_model_path = os.path.join(os.pardir, "models", full_model_name + "_best.h5")
    plot_path = os.path.join(os.pardir, "models", full_model_name + ".png")
    history_path = os.path.join(os.pardir, "models", "history", full_model_name + "_history.npy")

    # Try loading the model, ...
    try:

        autoencoder = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path)
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except:

        log("Training linear autoencoder.", lvl=2)

        # Flatten
        x_tr = squash(x_tr)
        x_va = squash(x_va)
        input_shape = (image_size,)

        # Build model
        autoencoder = Sequential()
        autoencoder.add(Dense(encoding_dim, input_shape=input_shape, activation='linear',
                              kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(Dense(image_size, activation='linear',
                              kernel_initializer='random_uniform', bias_initializer='zeros'))

        # Compile model
        autoencoder.compile(optimizer=optimizer, loss=loss)

        # Training parameters
        es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        mc = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        tb = TensorBoard(log_dir='/tmp/auto_lin')

        # Train model
        history = autoencoder.fit(x_tr, x_tr,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_va, x_va),
                        callbacks=[es, mc, tb])

        # Save model and history
        autoencoder.save(model_path)
        np.save(file=history_path, arr=history)

        # Visual aid
        plot_model(autoencoder, to_file=plot_path, show_layer_names=True, show_shapes=True)

    # Get intermediate output at encoded layer
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=0).output)

    # See effect of encoded representations on output (eigenfaces?)
    '''decoder = Model(inputs=Input(shape=(encoding_dim,)), outputs=autoencoder.output)
    plot_model(autoencoder, to_file=os.path.join(os.pardir, "models", "decoder.png"), show_layer_names=True, show_shapes=True)'''

    # Size of encoded representation
    compression_factor = np.round(float(image_size / encoding_dim), decimals=2)
    log("Compression factor is {}".format(compression_factor), lvl=3)

    return autoencoder, encoder, history


def deep_cnn_autoencoder(x_tr: np.ndarray, x_va: np.ndarray,
                         model_name='dcnn_autoencoder',
                         encoding_dim=32, epochs=75,
                         optimizer="adam", loss="mean_squared_error") -> \
        (Sequential, Sequential, float):
    """
    Build a deep convolutional autoencoder.

    :param model_name: name used in file creation
    :param x_tr: training images
    :param x_va: validation images
    :param encoding_dim: number of nodes in encoder layer (i.e. the bottleneck)
    :param epochs: number of epochs to train for
    :param optimizer: optimizer to use in training
    :param loss: loss function to use in training
    :return: autoencoder model, encoder model, decoder model, compression factor
    """

    # Get image dimension
    image_dim = x_tr.shape[1]
    image_size = image_dim ** 2 * 3

    # Set parameters
    batch_size = 32

    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-en_dim' + str(encoding_dim)

    # Build model path
    model_path = os.path.join(os.pardir, "models", full_model_name + ".h5")
    best_model_path = os.path.join(os.pardir, "models", full_model_name + "_best.h5")
    plot_path = os.path.join(os.pardir, "models", full_model_name + ".png")
    history_path = os.path.join(os.pardir, "models", "history", full_model_name + "_history.npy")

    # Try loading the model, ...
    try:

        autoencoder = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path)
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except:

        log("Training deep convolutional autoencoder.", lvl=2)

        # Input shape: image dim x image dim x 3
        input_shape = x_tr.shape[1:]

        # Build model
        autoencoder = Sequential()
        autoencoder.add(Conv2D(2 * encoding_dim, (10, 10), padding='same', activation='relu', input_shape=input_shape,
                               kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(MaxPooling2D((2, 2), padding='same'))

        autoencoder.add(Conv2D(encoding_dim, (10, 10), padding='same', activation='relu',
                               kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(MaxPooling2D((2, 2), padding='same'))

        autoencoder.add(Conv2D(encoding_dim, (10, 10), padding='same', activation='relu',
                               kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(UpSampling2D((2, 2)))

        autoencoder.add(Conv2D(2 * encoding_dim, (10, 10), padding='same', activation='relu',
                               kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(BatchNormalization())
        autoencoder.add(UpSampling2D((2, 2)))

        autoencoder.add(Conv2D(3, (10, 10), padding='same', activation='sigmoid',
                               kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(BatchNormalization())

        # Compile model
        autoencoder.compile(optimizer=optimizer, loss=loss)

        # Training parameters
        es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        mc = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        tb = TensorBoard(log_dir='/tmp/auto_dcnn')

        # Train model
        history = autoencoder.fit(x_tr, x_tr,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_va, x_va),
                        callbacks=[es, mc, tb])

        # Save model and history
        autoencoder.save(model_path)
        np.save(file=history_path, arr=history)

        # Visual aid
        plot_model(autoencoder, to_file=plot_path, show_layer_names=True, show_shapes=True)

    # Get intermediate output at encoded layer
    # encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='max_pooling2d_2').output)
    encoder = None

    # See effect of encoded representations on output (eigenfaces?)
    '''decoder = Model(inputs=Input(shape=(encoding_dim,)), outputs=autoencoder.output)
    plot_model(autoencoder, to_file=os.path.join(os.pardir, "models", "decoder.png"), show_layer_names=True, show_shapes=True)'''

    # Size of encoded representation
    compression_factor = np.round(float(image_size / encoding_dim), decimals=2)
    log("Compression factor is {}".format(compression_factor), lvl=3)

    return autoencoder, encoder, history


def plot_encoder_results(autoencoder: Sequential, convolutional: bool, model_name: str,
                         compression_factor: float, x: np.ndarray, examples=5, save=True):
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
        reconstructed_image = x_pred[col].reshape(image_dim, image_dim, 3)

        ax = subplot_axes[row][col]
        if row == 0:
            ax.set_title("Original")
            ax.imshow(original_image)
        else:
            ax.set_title("Reconstructed")
            ax.imshow(reconstructed_image)

        ax.axis('off')

    # General make-up
    sup_string = ('Deep CNN' if convolutional else 'Linear') +\
                 ' autoencoder - image dim {}, compression factor {}'.format(image_dim, compression_factor)
    plt.suptitle(sup_string,
                 fontweight='bold', fontsize='12')
    plt.tight_layout()

    # Save Figure
    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-en_dim' + str(encoding_dim)

    # Build model path
    model_path = os.path.join(os.pardir, "models", "plots", full_model_name + ".png")
    if save: plt.savefig(model_path)
    plt.show()


if __name__ == "__main__":
    """Main function."""

    # Set parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Loop over these dimensions
    image_dims = (32, 64, 128)
    encoding_dims = (128, 256, 512)

    for image_dim in image_dims:

        for encoding_dim in encoding_dims:
            # Calculate extra parameters
            image_size = image_dim ** 2 * 3
            compression_factor = np.round(float(image_size / encoding_dim), decimals=2)

            # Check folders
            make_folders('models',
                         os.path.join('models', 'plots'),
                         os.path.join('models', 'history'))

            # Get the data
            (x_tr, y_tr), (x_va, y_va) = dat.pipeline(image_dim=image_dim)

            # Build linear autoencoder
            lin_auto, lin_enc, lin_hist = linear_autoencoder(model_name='linear_autoencoder',
                                                   x_tr=x_tr, x_va=x_va,
                                                   encoding_dim=encoding_dim, epochs=80)
            dcnn_auto, dcnn_enc, lin_hist = deep_cnn_autoencoder(x_tr=x_tr, x_va=x_va,
                                                       model_name='dcnn_autoencoder',
                                                       encoding_dim=encoding_dim,
                                                       epochs=75)

            # Check results
            plot_encoder_results(model_name='linear_autoencoder', autoencoder=lin_auto, convolutional=False,
                                 compression_factor=compression_factor,
                                 x=x_va)
            plot_encoder_results(model_name='dcnn_autoencoder', autoencoder=dcnn_auto, convolutional=True,
                                 compression_factor=compression_factor,
                                 x=x_va)
