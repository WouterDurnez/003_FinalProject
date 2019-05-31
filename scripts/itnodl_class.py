#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
TASK 3 - CLASSIFIER

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, History
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve, hamming_loss, accuracy_score
from pprint import PrettyPrinter

import itnodl_help as hlp
from itnodl_auto import build_autoencoder
from itnodl_data import pipeline
from itnodl_help import log, set_up_model_directory


def build_classifier(image_dim: int, compression_factor: int,
                     x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
                     from_scratch=False, all_trainable=False, epochs=200, patience=25) -> (Sequential, History):
    """
    Build a deep convolutional classifier network, either from scratch, or from a pretrained autoencoder.

    :param image_dim: dimension of training and validation images
    :param compression_factor: size of bottleneck in network
    :param x_tr: training images
    :param y_tr: training labels
    :param x_va: validation images
    :param y_va: validation labels
    :param from_scratch: do we start fresh, or from earlier weights?
    :param all_trainable: are all weights trainable, or do we freeze the encoder part?
    :param epochs: number of epochs to train for
    :param patience: number of epochs to wait before we stop training, when seeing no improvement in validation loss
    :return: classifier model, training history
    """

    # Model parameters
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    # Set verbosity
    if hlp.LOG_LEVEL == 3:
        verbose = 1
    elif hlp.LOG_LEVEL == 2:
        verbose = 2
    else:
        verbose = 0

    # Start building new Sequential model
    classifier = Sequential()

    # Use previously trained model (we could just load weights, too).
    if not from_scratch:

        # Build encoder
        _, encoder, _ = build_autoencoder(model_name='conv_auto',
                                          convolutional=True,
                                          train=False,
                                          x_tr=x_tr, x_va=x_va,
                                          compression_factor=compression_factor)

        # Freeze, dirtbag (or not, I'm not dirty Harry)
        conv_layers = [1, 2, 4, 5]
        for i in conv_layers:
            encoder.get_layer(index=i).trainable = all_trainable

        # Add encoding part of autoencoder to our classifier
        classifier.add(encoder)

    # ... unless we just use the architecture
    else:

        input_shape = (image_dim, image_dim, 3)

        classifier.add(Conv2D(2 * encoding_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                              kernel_initializer='random_uniform', bias_initializer='zeros'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D((2, 2), padding='same'))

        classifier.add(Conv2D(encoding_dim, (3, 3), padding='same', activation='relu',
                              kernel_initializer='random_uniform', bias_initializer='zeros'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

    # Add classification layers
    classifier.add(Flatten())
    classifier.add(Dense(encoding_dim))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(5, activation='sigmoid'))  # <-- multilabel (for multiclass: softmax)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Data augmentation to get the most out of our images
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_tr)

    # Callbacks
    if patience == 0:
        patience = epochs
    es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
    mc = ModelCheckpoint(filepath=classifier_path, monitor='val_loss', verbose=verbose, save_best_only=True)
    tb = TensorBoard(log_dir='/tmp/{}_im{}comp{}_full{}'.format(classifier_name,
                                                                image_dim,
                                                                compression_factor,
                                                                all_trainable))

    # Train model using data augmentation
    history = classifier.fit_generator(datagen.flow(x_tr, y_tr, batch_size=32),
                                       epochs=epochs,
                                       steps_per_epoch=x_tr.shape[0] // 32,
                                       verbose=verbose,
                                       validation_data=(x_va, y_va),
                                       callbacks=[es, mc, tb])
    '''history = classifier.fit(x=x_tr, y=y_tr,
                             batch_size=32,
                             epochs=epochs,
                             verbose=verbose,
                             callbacks=[es, mc, tb],
                             validation_data=(x_va, y_va))'''

    # Save model and history
    classifier.save(classifier_path)

    return classifier, history


def plot_model_metrics(histories: dict) -> plt.Axes:
    """
    Plot the history of the model metrics.

    :param histories: histories of trained models
    :return: plot axes
    """

    # Set style
    sns.set_style('whitegrid')

    # Initialize axes
    fig, subplot_axes = plt.subplots(2, 2,
                                     squeeze=False,
                                     sharex='none',
                                     sharey='none',
                                     figsize=(10, 10))

    # Fill axes
    for col in range(2):

        train_or_val = 'Train' if col == 0 else 'Validation'

        for row in range(2):

            ax = subplot_axes[row][col]

            for label, history in histories.items():

                if row == 0:
                    key = 'acc' if col == 0 else 'val_acc'
                    title = '{} accuracy'.format(train_or_val)
                    y_label = 'Accuracy'
                    y_limit = (.7, .9)

                else:
                    key = 'loss' if col == 0 else 'val_loss'
                    title = '{} loss'.format(train_or_val)
                    y_label = 'Loss (binary cross entropy)'
                    y_limit = (.2, .7)

                # Plot training & validation accuracy values

                ax.plot(history.history[key], label="Model: {}".format(label))

                ax.set_title(title, fontdict={'fontweight': 'bold'})
                ax.set_ylabel(y_label)
                ax.set_ylim(y_limit)
            ax.set_xlabel('Epoch')
            ax.set_xlim(0, 100)
            ax.legend(loc='auto')

    plt.show()

    return ax


def evaluate_classifier(classifier: Model, x: np.ndarray, y: np.ndarray, threshold=.5) -> dict:
    # Store metrics in dictionary
    metrics = {}

    # Get probabilities
    y_prob = classifier.predict(x)

    # ... and extract for predictions
    y_pred = y_prob
    super_threshold_indices = y_prob > threshold
    y_pred[super_threshold_indices] = 1
    y_pred[np.invert(super_threshold_indices)] = 0

    pp = PrettyPrinter(indent=5)

    metrics['Hamming loss'] = hamming_loss(y, y_pred)
    metrics['Exact match ratio'] = accuracy_score(y, y_pred)

    pp.pprint(metrics)

    return metrics


if __name__ == "__main__":

    # Let's go
    log("CLASSIFIERS", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('classifiers')

    # Set model parameters
    autoencoder_name = 'conv_auto'
    classifier_name = 'conv_class'
    image_dim = 32
    image_size = image_dim ** 2 * 3
    compression_factor = 32
    encoding_dim = image_size // compression_factor

    # Get the data
    (x_tr, y_tr), (x_va, y_va) = pipeline(image_dim=image_dim)

    # Store histories
    histories = {}

    # Model parameters to loop over
    parameter_combinations = [(False, False),
                              (False, True),
                              (True, True)]

    # Compare classifiers
    for (from_scratch, all_trainable) in parameter_combinations:
        # Full model name for file output
        full_autoencoder_name = "{}_im_dim{}-comp{}".format(autoencoder_name, image_dim, compression_factor)
        full_classifier_name = "{}_im_dim{}-comp{}_full{}_scratch{}".format(classifier_name, image_dim,
                                                                            compression_factor,
                                                                            all_trainable, from_scratch)

        # Build paths
        autoencoder_path = os.path.join(os.pardir, "models", "autoencoders", full_autoencoder_name + ".h5")
        classifier_path = os.path.join(os.pardir, "models", "classifiers", full_classifier_name + ".h5")
        history_path = os.path.join(os.pardir, "models", "classifiers", "history",
                                    full_classifier_name + "_history.npy")

        history_label = "{}from scratch, {}all trainable".format(("" if from_scratch else "not "),
                                                                 ("" if all_trainable else "not "))

        # Build classifier
        classifier, histories[history_label] = build_classifier(image_dim=image_dim,
                                                                compression_factor=compression_factor,
                                                                x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va,
                                                                from_scratch=from_scratch, all_trainable=all_trainable,
                                                                epochs=100, patience=0)

        # Save history
        np.save(file=history_path, arr=histories[history_label])

    # Plot model metrics
    plot_model_metrics(histories=histories)

    '''Further evaluation'''

    # TODO: need better metrics here. Accuracy, exact match ration, precision, recall, Hamming loss?

    # Compare classifiers
    y_va_pre = classifier.predict(x_va)
    y_tr_pre = classifier.predict(x_tr)

    for threshold in np.arange(.1, .9, .05):
        print(threshold)
        evaluate_classifier(classifier, x=x_va, y=y_va, threshold=threshold)

    correct = 0
    for i in range(len(y_va)):

        equal = np.equal(y_va[i], np.round(y_va_pre[i], 0))
        all = equal.all()

        if all:
            correct += 1

        print("{} -> {} -- {} ({})".format(y_va[i], np.round(y_va_pre[i], 0), equal, all))

    correct2 = 0
    for i in range(len(y_va)):

        equal = np.equal(y_va[i], np.round(y_va_pre[i], 0))
        all = equal.all()

        if all:
            correct2 += 1

        print("{} -> {} -- {} ({})".format(y_va[i], np.round(y_va_pre[i], 0), equal, all))
