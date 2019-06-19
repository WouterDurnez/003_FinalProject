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
import random as rnd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from skimage.color import gray2rgb

import itnodl_data as dat
import itnodl_help as hlp
import itnodl_segm_unet as unet
from itnodl_data import threshold
from itnodl_help import log, set_up_model_directory


#############
# Dice loss # (All scavenged from the internet)
#############

def dice_coef(y_true, y_pred) -> float:  # Taken from https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    dice = np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


#################
# Architectures #
#################

def convolutional_segm_architecture(image_dim: int, optimizer: str, loss: str, encoding_dim: int, mono=True,
                                    auto=False) -> Model():
    """
    Build architecture for deep convolutional autoencoder.

    :param image_dim: dimension of image (total size = image dim x image dim x color channels)
    :param optimizer: optimizer function
    :param loss: loss function
    :return: segmentation network architecture
    """

    # Model parameters
    input_shape = (image_dim, image_dim, 3)
    image_size = np.prod(input_shape)

    compression_factor = image_size // encoding_dim

    # Build model
    segnet = Sequential()

    if auto:
        segnet.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(MaxPooling2D((2, 2), padding='same'))

        segnet.add(Conv2D(image_dim // (2 * compression_factor), (3, 3), padding='same', activation='relu',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

        segnet.add(Conv2D(image_dim // (2 * compression_factor), (3, 3), padding='same', activation='relu',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(UpSampling2D((2, 2)))

        segnet.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(UpSampling2D((2, 2)))
    else:
        segnet.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(MaxPooling2D((2, 2), padding='same'))

        segnet.add(Conv2D(4 * image_dim // compression_factor, (3, 3), padding='same', activation='relu',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

        segnet.add(Conv2D(4 * image_dim // compression_factor, (3, 3), padding='same', activation='relu',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
        segnet.add(BatchNormalization())
        segnet.add(UpSampling2D((2, 2)))

        segnet.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu',
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
                   patience=5,
                   auto=False) -> (Sequential, History):
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
    batch_size = 128
    if hlp.LOG_LEVEL == 3:
        verbose = 1
    elif hlp.LOG_LEVEL == 2:
        verbose = 2
    else:
        verbose = 0

    # Full model name for file output
    full_model_name = "{}_im_dim{}-comp{}_{}_mono{}".format(model_name, image_dim, compression_factor, loss, mono)

    if auto:
        full_model_name += "auto"

    # Build model path
    model_path = os.path.join(os.pardir, "models", "segmentation", full_model_name + ".h5")
    architecture_path = os.path.join(os.pardir, "models", "segmentation", "architecture",
                                     full_model_name + "_architecture.png")

    # Keep track of history
    history_path = os.path.join(os.pardir, "models", "segmentation", "history", full_model_name + "_history.npy")

    # Try loading the model, ...
    try:

        segnet = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path).tolist()
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except Exception as e:

        # log(e)

        log("Building segmentation network.")
        segnet = convolutional_segm_architecture(image_dim=image_dim, optimizer=optimizer, loss=loss,
                                                 encoding_dim=encoding_dim, auto=auto)

        # Print model info
        log("Network parameters: image dimension {}, image size {}, compression factor {}.".
            format(image_dim, image_size, compression_factor), lvl=3)

        # Train the model (either continue training the old model, or train the new one)
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


#############
# Visualize #
#############

def mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_res = x.copy()
    for color_channel in range(3):
        y_res[:, :, :, color_channel] = np.multiply(x[:, :, :, color_channel],
                                                    threshold(y, threshold=.5, mono=False)[:, :, :, 0])

    return y_res


def plot_segmentation_results(image_list: list, labels: list, examples=5, random=True, seed=818, save=True, plot=True,
                              name="", indices=None):
    """

    """

    # Set font
    font = {'fontname': 'Times New Roman Bold',
            'fontfamily': 'serif',
            'weight': 'bold'}

    image_dim = image_list[0].shape[1]
    print(image_dim)

    # Indices
    if indices == None:
        if random:
            rnd.seed(seed)
            indices = rnd.sample(range(image_list[0].shape[0]), k=examples)
        else:
            indices = [i for i in range(examples)]

    # Plot parameters
    row_count = len(image_list)
    col_count = examples
    plot_count = row_count * col_count

    # Initialize axes
    fig, subplot_axes = plt.subplots(row_count,
                                     col_count,
                                     squeeze=False,
                                     sharex='all',
                                     sharey='all',
                                     figsize=(12, 12))
    # Fill axes
    for i in range(plot_count):
        row = i // col_count
        col = i % col_count

        image = image_list[row][indices[col]]

        if image.shape[2] == 1:
            image = gray2rgb(image[:, :, 0])

        ax = subplot_axes[row][col]

        if col == 0:
            ax.set_ylabel(labels[row], fontdict=font)
        ax.imshow(image, zorder=1)

        # ax.axis('off')

    # General make-up
    plt.setp(subplot_axes, xticks=[], xticklabels=[],
             yticks=[])
    plt.tight_layout()

    # Save Figure
    # Full model name for file output
    full_result_name = 'seg_' + name + '_im_dim' + str(image_dim)

    # Build model path
    model_path = os.path.join(os.pardir, "models", "segmentation", "plots", full_result_name + ".png")

    # Show 'n tell
    if save: plt.savefig(model_path)
    if plot: plt.show()


def plot_model_history(history: History, image_dim: int, model_type="segmentation", save=True, plot=True) -> plt.Axes:
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
    colors = sns.color_palette('pastel', 2)

    # Initialize axes
    fig = plt.figure(figsize=(8, 5), dpi=150)
    ax = plt.axes()

    # Fill axes
    n_epochs = len(history.history['loss'])

    y_label = 'Loss (mean squared error)'

    # Plot training & validation accuracy values
    ax.plot(history.history["loss"], label="Training", color=colors[0])
    ax.plot(history.history["val_loss"], label="Validation", color=colors[1])

    # Add vertical line to indicate early stopping
    ax.axvline(x=n_epochs - 1, linestyle='--', color=colors[0])

    # Set a title, the correct y-label, and the y-limit
    ax.set_ylabel(y_label, fontdict={'fontname': 'Times New Roman'})
    ax.set_ylim(0.1, 1.2)
    ax.set_yscale("log")

    ax.set_xlabel('Epoch', fontdict={'fontname': 'Times New Roman'})
    ax.set_xlim(0, 500)

    ax.legend(loc='best', prop={'family': 'Serif'})

    # Build model path
    evaluation_label = 'histories_segm_im_dim{}'.format(image_dim)
    plot_path = os.path.join(os.pardir, "models", model_type, "plots", evaluation_label + ".png")

    # Show 'n tell
    if save: fig.savefig(plot_path, dpi=fig.dpi)
    if plot: plt.show()

    return fig


def plot_model_histories(histories: dict, image_dim: int, compression_factor: int, loss: str, save=True,
                         plot=True) -> plt.Axes:
    """
    Plot the histories of the model metrics 'loss' and 'accuracy'.

    :param histories: histories of trained models
    :param image_dim: image dimension
    :param compression_factor: (sic)
    :param save: save plot to png
    :param plot: show plot
    :return: plot axes
    """

    log("Plotting model metrics.", lvl=2)

    # Set style
    sns.set_style('whitegrid')
    colors = sns.color_palette('pastel', n_colors=len(histories))

    # Initialize axes
    fig, subplot_axes = plt.subplots(1, 2,
                                     squeeze=False,
                                     sharex='none',
                                     sharey='none',
                                     figsize=(11, 4),
                                     constrained_layout=True)
    fig.dpi = 150

    # Fill axes
    for col in range(2):

        train_or_val = 'Training' if col == 0 else 'Validation'

        ax = subplot_axes[0][col]

        color_counter = 0
        for label, history in histories.items():

            n_epochs = len(history.history['loss'])

            key = 'loss' if col == 0 else 'val_loss'
            title = '{} loss'.format(train_or_val)
            y_label = 'Loss (mean squared error)' if col == 0 else ""
            y_limit = (0, 1)

            # Plot label
            plot_label = "{}".format(label.capitalize())
            # Plot training & validation accuracy values
            ax.plot(history.history[key], label=plot_label, color=colors[color_counter])

            # Add vertical line to indicate early stopping
            ax.axvline(x=n_epochs - 1, linestyle='--', color=colors[color_counter])

            # Set a title, the correct y-label, and the y-limit
            ax.set_title(title, fontdict={'fontweight': 'semibold', 'family': 'serif'})
            ax.set_ylabel(y_label, fontdict={'family': 'serif'})
            ax.set_ylim(y_limit)

            color_counter += 1

            ax.set_yscale("log")

            if col == 0:
                ax.legend(loc='best', prop={'family': 'serif'})

            ax.set_xlabel('Epoch', fontdict={'family': 'serif'})
            # ax.set_xlim(0, 100)
            # fig.legend(loc='best', prop={'weight': 'bold', 'family':'serif'})

    # Title
    '''plt.suptitle(
        "Training histories of classifier models (image dim {} - compression {})".format(image_dim, compression_factor),
        fontweight='bold')'''

    # Build model path
    evaluation_label = 'histories_segm_im_dim{}comp{}loss{}'.format(image_dim, compression_factor, loss)
    plot_path = os.path.join(os.pardir, "models", "segmentation", "plots", evaluation_label + ".png")

    # Show 'n tell
    if save: fig.savefig(plot_path, dpi=fig.dpi)
    if plot: plt.show()

    return ax


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
    image_dim = 96
    compression_factor = 24
    train = False
    epochs = 500
    patience = 50
    loss = 'mean_squared_error'

    # Set remaining parameters
    segmentation_name = 'conv_segm'
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    # Get the data
    _, data = dat.pipeline(image_dim=image_dim, class_data=True, segm_data=True, mono=True)

    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']

    # Train segmentation network
    log("Home-made segmentation network", title=True)
    segnet, history = build_segm_net('conv_segm', train=train, x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va,
                                     compression_factor=compression_factor, epochs=epochs,
                                     optimizer='adam', loss=loss, patience=patience, mono=True, auto=False)

    x_te, y_te, y_te_src = data['x_test'], data['y_test'], data['y_test_source']

    y_tr_pred = segnet.predict(x=x_tr)
    y_te_pred = segnet.predict(x=x_te)
    y_va_pred = segnet.predict(x=x_va)

    # Visualize results
    log("Visualizing results.", lvl=1)
    y_te_res = x_te.copy()
    for color_channel in range(3):
        y_te_res[:, :, :, color_channel] = np.multiply(x_te[:, :, :, color_channel],
                                                       threshold(y_te_pred, threshold=.5, mono=False)[:, :, :, 0])

    plot_segmentation_results(image_list=[x_tr, y_tr, y_tr_pred, threshold(y_tr_pred, threshold=.5, mono=False)],
                              labels=["Original", "Target", "Predicted", "Threshold"],
                              examples=5, random=True, save=True, plot=True, name="train", seed=87)
    #  indices=[49, 109, 147]

    plot_segmentation_results(image_list=[x_va, y_va, y_va_pred, threshold(y_va_pred, threshold=.5, mono=False)],
                              labels=["Original", "Target", "Predicted", "Threshold"],
                              examples=5, random=True, save=True, plot=True, name="val")

    plot_segmentation_results(
        image_list=[x_te, y_te, y_te_pred, threshold(y_te_pred, threshold=.5, mono=False), y_te_res],
        labels=["Original", "Target", "Predicted", "Threshold", "Masked"],
        examples=5, random=True, save=True, plot=True, name="test")

    # Visualize histories
    plot_model_history(history=history, image_dim=image_dim, save=True, plot=True, model_type="Segmentation")

    # Evaluation
    log("Evaluating model.", lvl=1)
    segnet_eval = []
    segnet_dice = []

    for x, y in zip([x_tr, x_va, x_te], [y_tr, y_va, y_te]):
        segnet_eval.append(segnet.evaluate(x, y))
        segnet_dice.append(1 - dice_coef(y, segnet.predict(x)))

    # print(segnet_eval)
    #print(segnet_dice)

    # Compare with U-Net
    log("U-net-based segmentation network", title=True)
    unet_model, unet_history = unet.build_unet_segmentation_network(model_name='conv_segm_unet', x_tr=x_tr, y_tr=y_tr,
                                                                    x_va=x_va,
                                                                    y_va=y_va,
                                                                    epochs=epochs, optimizer='adam', loss=loss,
                                                                    patience=patience)

    plot_model_history(history=unet_history, image_dim=image_dim, save=True, plot=True, model_type="Segmentation")

    histories = {
        "Home-made": history,
        "U-Net-based": unet_history
    }

    plot_model_histories(histories=histories, image_dim=96, compression_factor=24,
                         loss='mean squared error', save=True, plot=True)

    #print(
