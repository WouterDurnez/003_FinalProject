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
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt

import scripts.itnodl_data as dat
import scripts.itnodl_help as hlp
from scripts.itnodl_help import log, make_folders


def prep_input(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing of input ndarray.

    :param x: array of indices (shape: n x image_size x image_size x 3)
    :return: mean corrected x
    """
    x_new = x[:, ] - mean
    return x_new


def linear_autoencoder_architecture(input_shape: tuple, optimizer='adam', loss='mean_squared_error'):

    # Build model
    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, input_shape=input_shape, activation='linear',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(Dense(image_size, activation='linear',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))

    # Build encoder part
    '''input_img = Input(shape=input_shape)
    encoder_layer = autoencoder.layers[0]
    encoder = Model(input_img, encoder_layer(input_img))'''

    # Train model
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder


def deep_cnn_autoencoder_architecture():

    # Build model
    autoencoder = Sequential()

    return autoencoder


def create_autoencoder(model_name: str, x_tr: np.ndarray, x_va: np.ndarray, encoding_dim=32, epochs=100) -> \
        (Sequential, Sequential, float):
    """
    Build a linear autoencoder.

    :param model_name: name used in file creation
    :param x_tr: training images
    :param x_va: validation images
    :param encoding_dim: number of nodes in encoder layer (i.e. the bottleneck)
    :return: autoencoder model, encoder model, decoder model, compression factor
    """

    # Get image dimension
    image_dim = x_tr.shape[1]
    image_size = image_dim**2 * 3

    # Set parameters
    batch_size = 32

    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-en_dim' + str(encoding_dim)

    # Build model path
    model_path = os.path.join(os.pardir, "models", full_model_name + ".h5")
    plot_path = os.path.join(os.pardir, "models", full_model_name + ".png")

    # Try loading the model, ...
    try:

        autoencoder = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)

    # ... otherwise, create it
    except:

        log("Training linear autoencoder.", lvl=2)

        # Flatten
        x_tr = x_tr.reshape((len(x_tr), image_size))
        x_va = x_va.reshape((len(x_va), image_size))
        input_shape = (image_size,)

        # Build model
        # Build model
        autoencoder = Sequential()
        autoencoder.add(Dense(encoding_dim, input_shape=input_shape, activation='linear',
                              kernel_initializer='random_uniform', bias_initializer='zeros'))
        autoencoder.add(Dense(image_size, activation='linear',
                              kernel_initializer='random_uniform', bias_initializer='zeros'))

        # Build encoder part
        '''input_img = Input(shape=input_shape)
        encoder_layer = autoencoder.layers[0]
        encoder = Model(input_img, encoder_layer(input_img))'''

        # Train model
        autoencoder.compile(optimizer=optimizer, loss=loss)
        # Train model
        autoencoder.fit(x_tr, x_tr,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_va, x_va),
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

        # Save model
        autoencoder.save(model_path)

        # Visual aid
        plot_model(autoencoder, to_file=plot_path, show_layer_names=True, show_shapes=True)

    # Get intermediate output at encoded layer
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=0).output)

    # See effect of encoded representations on output (eigenfaces?)
    decoder = Model(inputs=Input(shape=(encoding_dim,)), outputs=autoencoder.output)
    plot_model(autoencoder, to_file=os.path.join(os.pardir, "models", "decoder.png"), show_layer_names=True, show_shapes=True)

    # Size of encoded representation
    compression_factor = np.round(float(image_size / encoding_dim), decimals=2)
    log("Compression factor is {}".format(compression_factor), lvl=3)

    return autoencoder, encoder, decoder, compression_factor


def plot_encoder_results(autoencoder: Sequential, model_name: str, compression_factor: float, x: np.ndarray, examples=5,
                         save=True):
    """
    Plot a number of examples: compare original and reconstructed images.

    :param autoencoder: the model to evaluate
    :param compression_factor: the model's compression factor
    :param x: the data to pass through
    :param examples: number of images to visualize
    :return:
    """

    # Get image dimension
    image_dim = x.shape[1]

    # Check results
    x_pred = autoencoder.predict(x=x.reshape(len(x), image_size))

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
    plt.suptitle('Linear autoencoder - image dim {}, compression factor {}'.format(image_dim, compression_factor),
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
    image_dim = 128
    image_size = image_dim ** 2 * 3
    encoding_dim = 150
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    make_folders('models', os.path.join('models', 'plots'))

    # Get the data
    (x_tr, y_tr), (x_va, y_va) = dat.pipeline(image_dim=image_dim)

    # Build linear autoencoder
    autoencoder, encoder, decoder, compression_factor = \
        create_autoencoder(model_name='autoencoder',
                           x_tr=x_tr, x_va=x_va,
                           encoding_dim=encoding_dim,
                           epochs=80)

    # Check results
    plot_encoder_results(model_name='autoencoder', autoencoder=autoencoder, compression_factor=compression_factor,
                         x=x_va)
