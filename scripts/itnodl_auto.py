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
import random as rnd
from math import ceil

import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model, load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt

import itnodl_data as dat
import itnodl_help as hlp
from itnodl_data import squash, lift, show_some
from itnodl_help import log, set_up_model_directory


#################
# Architectures #
#################

def linear_auto_architecture(image_size: int, optimizer: str, loss: str, compression_factor: int) -> Model():
    """
    Build architecture for linear autoencoder with single layer.

    :param image_size: data size (image dim x image dim x color channels)
    :param optimizer: optimizer function
    :param loss: loss function
    :return:
    """

    # Model parameters
    input_shape = (image_size,)
    encoding_dim = image_size // compression_factor

    # Build model
    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, input_shape=input_shape, activation='linear',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(Dense(image_size, activation='linear',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))

    # Compile model
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return autoencoder


def convolutional_auto_architecture(image_dim: int, optimizer: str, loss: str, compression_factor: int) -> Model():
    """
    Build architecture for deep convolutional autoencoder.

    :param image_dim: dimension of image (total size = image dim x image dim x color channels)
    :param optimizer: optimizer function
    :param loss: loss function
    :return:
    """

    # Model parameters
    input_shape = (image_dim, image_dim, 3)
    image_size = np.prod(input_shape)

    # Build model
    autoencoder = Sequential()
    autoencoder.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))

    autoencoder.add(Conv2D(image_dim // compression_factor, (3, 3), padding='same', activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

    autoencoder.add(Conv2D(image_dim // compression_factor, (3, 3), padding='same', activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(UpSampling2D((2, 2)))

    autoencoder.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(UpSampling2D((2, 2)))

    autoencoder.add(Conv2D(3, (10, 10), padding='same', activation='sigmoid',
                           kernel_initializer='random_uniform', bias_initializer='zeros'))
    autoencoder.add(BatchNormalization())

    # Compile model
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()
    return autoencoder


###################
# Build and train #
###################

def build_autoencoder(model_name: str, convolutional: bool, train: bool,
                      x_tr: np.ndarray, x_va: np.ndarray,
                      compression_factor: int, epochs=100,
                      optimizer="adam", loss="mean_squared_error",
                      patience=5) -> \
        (Sequential, Sequential, float):
    """
    Build an autoencoder.

    :param model_name: name used in file creation
    :param x_tr: training images
    :param x_va: validation images
    :param compression_factor: what's the size of the bottleneck?
    :param epochs: number of epochs to train for
    :param optimizer: optimizer to use in training
    :param loss: loss function to use in training
    :param patience: number of epochs without improvement before early stop
    :return: autoencoder model, encoder model, decoder model, compression factor
    """

    # Model parameters
    image_dim = x_tr.shape[1]
    image_size = image_dim ** 2 * 3

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
            log("Building deep convolutional autoencoder.", lvl=2)
            autoencoder = convolutional_auto_architecture(image_dim=image_dim, optimizer=optimizer, loss=loss,
                                                          compression_factor=compression_factor)
        else:
            log("Building linear autoencoder.", lvl=2)
            autoencoder = linear_auto_architecture(image_size=image_size, optimizer=optimizer, loss=loss,
                                                   compression_factor=compression_factor)

        # Print model info
        log("Network parameters: image dimension {}, image size {}, compression factor {}.".
            format(image_dim, image_size, compression_factor), lvl=3)

    # Train the model (either continue training the old model, or train the new one)
    if train:

        log("Training autoencoder.", lvl=2)
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

    # Set font
    font = {'fontname': 'Times New Roman Bold',
            'fontfamily': 'serif',
            'weight': 'bold'}

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
            ax.set_title("Original", fontdict=font)
            ax.imshow(original_image)
        else:
            ax.set_title("Reconstructed", fontdict=font)
            ax.imshow(reconstructed_image)

        ax.axis('off')

    # General make-up
    sup_string = ('Deep CNN' if convolutional else 'Linear') + \
                 ' autoencoder - image dim {}, compression factor {}'.format(image_dim, compression_factor)
    '''plt.suptitle(sup_string,
                 fontweight='bold', fontsize='12')'''
    plt.tight_layout()

    # Save Figure
    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))

    # Build model path
    model_path = os.path.join(os.pardir, "models", "autoencoders", "plots", full_model_name + ".png")

    # Show 'n tell
    if save: plt.savefig(model_path)
    if plot: plt.show()


def plot_both_autoencoder_results(image_list: list, examples=5, random=True, save=True, plot=True):
    """

    """

    # Set font
    font = {'fontname': 'Times New Roman Bold',
            'fontfamily': 'serif',
            'weight': 'bold'}

    # Indices
    indices = rnd.sample(range(image_list[0].shape[0]), k=examples)

    # Plot parameters
    row_count = len(image_list)
    col_count = examples
    plot_count = row_count * col_count

    # Image lists
    x = image_list[0]
    x_pred_lin = image_list[1]
    x_pred_cnn = image_list[2]

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

        ax = subplot_axes[row][col]
        if row == 0:
            if col == 0: ax.set_ylabel("Original", fontdict=font)
            ax.imshow(x[indices[col]], zorder=1)
        elif row == 1:
            if col == 0: ax.set_ylabel("Linear", fontdict=font)
            ax.imshow(x_pred_lin[indices[col]], zorder=1)
        else:
            if col == 0: ax.set_ylabel("Convolutional", fontdict=font)
            ax.imshow(x_pred_cnn[indices[col]], zorder=1)

        # ax.axis('off')

    # General make-up
    plt.setp(subplot_axes, xticks=[], xticklabels=[],
             yticks=[])
    plt.tight_layout()

    # Save Figure
    # Full model name for file output
    full_result_name = 'reconstruction' + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))

    # Build model path
    model_path = os.path.join(os.pardir, "models", "autoencoders", "plots", full_result_name + ".png")

    # Show 'n tell
    if save: plt.savefig(model_path)
    if plot: plt.show()


def plot_model_histories(histories: dict, image_dim: int, compression_factor: int, save=True, plot=True) -> plt.Axes:
    """
    Plot the histories of the model metrics 'loss' and 'val loss'.

    :param histories: histories of trained models
    :param image_dim: image dimension
    :param compression_factor: (sic)
    :param save: save plot to png
    :param plot: show plot
    :return: plot axes
    """

    log("Plotting model metrics.", lvl=2)

    # Set font
    font = {'fontname': 'Times New Roman Bold',
            'fontfamily': 'serif',
            'weight': 'bold'}

    # Set style
    sns.set_style('whitegrid')
    colors = sns.color_palette('pastel', n_colors=len(histories))

    # Initialize axes
    fig, subplot_axes = plt.subplots(1, 2,
                                     squeeze=False,
                                     sharex='none',
                                     sharey='none',
                                     figsize=(10, 4),
                                     constrained_layout=True)

    # Fill axes
    for col in range(2):

        train_or_val = 'Training' if col == 0 else 'Validation'

        ax = subplot_axes[0][col]

        color_counter = 0
        for label, history in histories.items():
            n_epochs = len(history.history['loss'])

            key = 'loss' if col == 0 else 'val_loss'
            title = '{} loss'.format(train_or_val)
            y_label = 'Loss (mean squared error)' if col == 0 else ''
            y_limit = (0, .7)

            # Plot training & validation accuracy values
            ax.plot(history.history[key], label="Model: {}".format(label), color=colors[color_counter])

            # Add vertical line to indicate early stopping
            ax.axvline(x=n_epochs, linestyle='--', color=colors[color_counter])

            # Set a title, the correct y-label, and the y-limit
            ax.set_title(title, fontdict=font)
            ax.set_ylabel(y_label, fontdict={'fontname': 'Times New Roman'})
            ax.set_ylim(y_limit)
            ax.set_yscale("log")

            color_counter += 1

            ax.set_xlabel('Epoch', fontdict={'fontname': 'Times New Roman'})
            ax.set_xlim(0, 100)

            ax.legend(loc='best', prop={'family': 'Serif'})

    # Title
    '''plt.suptitle(
    "Training histories of classifier models (image dim {} - compression {})".format(image_dim, compression_factor),
    fontweight='bold')'''

    # Build model path
    evaluation_label = 'histories_im_dim{}comp{}'.format(image_dim, compression_factor)
    plot_path = os.path.join(os.pardir, "models", "autoencoders", "plots", evaluation_label + ".png")

    # Show 'n tell
    if save: fig.savefig(plot_path, dpi=fig.dpi)
    if plot: plt.show()

    return ax


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
    '''image_dims = (48,)
    compression_factors = (48,)'''

    # dims = [(i, c) for i in image_dims for c in compression_factors]

    image_dim, compression_factor = 64, 24

    # Calculate extra parameters
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    print("image size:", image_size,
          "encoding dim", encoding_dim)

    # Get the data
    data, _ = dat.pipeline(image_dim=image_dim)
    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']
    x_te, y_te = data['x_test'], data['y_test']

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
                                                        epochs=100,
                                                        patience=10)

        # Visualize
        plot_autoencoder_results(model_name='lin_auto', autoencoder=lin_auto, convolutional=False,
                                 compression_factor=compression_factor,
                                 x=x_te, save=True, plot=False)

        # Show error
        '''x_te_pred = lin_auto.predict(x=squash(x_te))
        lin_error = mean_squared_error(squash(x_te), x_te_pred)'''

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
                                 x=x_te, save=True, plot=True)

        # Show error
        '''x_te_pred = dcnn_auto.predict(x=x_te)
        conv_error = mean_squared_error(x_te, x_te_pred)'''

    histories = {
        'Linear autoencoder': lin_hist[0],
        'Deep convolutional autencoder': dcnn_hist[0]
    }

    # Visualization
    plot_model_histories(histories=histories, image_dim=image_dim,
                         compression_factor=compression_factor,
                         save=True, plot=True)

    x_te_pred_cnn = dcnn_auto.predict(x=x_te)
    x_te_pred_lin = lift(lin_auto.predict(x=squash(x_te)))

    plot_both_autoencoder_results(image_list=[x_te, x_te_pred_lin, x_te_pred_cnn],
                                  examples=5,
                                  save=True, plot=True)

    show_some(image_list=[x_te, x_te_pred_lin, x_te_pred_cnn],
              subtitles=["Original", "Linear autoencoder reconstruction",
                         "Deep convolutional autoencoder reconstruction"],
              n=5, random=True, save=True, save_name="auto_comparison")

    # Evalution
    lin_eval = lin_auto.evaluate(squash(x_te), squash(x_te))
    dcnn_eval = dcnn_auto.evaluate(x_te, x_te)

    # TSNE attempt 2
    '''x_latent = lin_enc.predict(x=squash(x_te))
    y_latent = [np.where(r == 1)[0][0] for r in y_te]
    labels = [dat.CLASSES[i] for i in y_latent]
    Y = tsne.tsne(x_latent, 3, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, y_latent)
    pylab.show()
    pylab.scatter(Y[:, 0], Y[:, 2], 20, y_latent)
    pylab.show()'''
