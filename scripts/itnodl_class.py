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
import random as rnd
from math import ceil
from pprint import PrettyPrinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, History
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, BatchNormalization, MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_similarity_score

import itnodl_data as dat
import itnodl_help as hlp
from itnodl_auto import build_autoencoder
from itnodl_help import log, set_up_model_directory


###################
# Build and train #
###################

def build_classifier(image_dim: int, compression_factor: int,
                     x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
                     from_scratch=False, all_trainable=False,
                     loss='binary_crossentropy', epochs=200, patience=25) -> (Sequential, History):
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

        image_size = np.prod(input_shape)

        # Build model
        classifier = Sequential()
        classifier.add(Conv2D(image_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                              kernel_initializer='random_uniform', bias_initializer='zeros'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D((2, 2), padding='same'))

        classifier.add(Conv2D(image_dim // compression_factor, (3, 3), padding='same', activation='relu',
                              kernel_initializer='random_uniform', bias_initializer='zeros'))
        classifier.add(BatchNormalization())
        classifier.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

    # Add classification layers
    classifier.add(Flatten())
    classifier.add(Dense(encoding_dim))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(5, activation='sigmoid'))  # <-- multilabel (for multiclass: softmax)
    classifier.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    # Plot model
    full_classifier_name = "{}_im_dim{}-comp{}_full{}_scratch{}_loss{}".format(classifier_name, image_dim,
                                                                               compression_factor,
                                                                               all_trainable, from_scratch,
                                                                               loss)
    architecture_path = os.path.join(os.pardir, "models", "classifiers", "architecture", full_classifier_name + ".png")
    plot_model(classifier, to_file=architecture_path, show_layer_names=True, show_shapes=True)

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
    tb = TensorBoard(log_dir='/tmp/{}_class_im{}comp{}_full{}_scratch{}'.format(classifier_name,
                                                                                image_dim,
                                                                                compression_factor,
                                                                                all_trainable,
                                                                                from_scratch))
    # Print model info
    log("Network parameters: image dimension {}, image size {}, compression factor {}.".
        format(image_dim, image_size, compression_factor), lvl=3)

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


############
# Evaluate #
############

def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case.
    '''

    # Store accuracy of each prediction
    accuracies = []

    for i in range(len(y_true)):

        set_true = set(np.where(y_true[i])[0])  # Positions of true labels
        set_pred = set(np.where(y_pred[i])[0])  # Positions of labels predicted to be true

        if len(set_true) == 0 and len(set_pred) == 0:
            accuracy_temp = 1
        else:
            accuracy_temp = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))

        accuracies.append(accuracy_temp)

    return float(np.mean(accuracies))


def label_based_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracies = []
    for i in range(len(y_true)):
        accuracy = np.sum(y_true[i] == y_pred[i]) / len(y_true[i])
        accuracies.append(accuracy)

    return float(np.mean(accuracies))


def evaluate_classifier(classifier: Model, x: np.ndarray, y: np.ndarray, threshold=.5) -> dict:
    """
    Evaluate classifier on number of metrics.

    :param classifier: the model that is to be evaluated.
    :param x: training (input) data which will lead to predictions
    :param y: expected outcomes
    :param threshold: boundary for prediction value to be considered 0 or 1
    :return:
    """

    log("Evaluating classifier.", lvl=2)

    # Store metrics in dictionary
    metrics = {}

    # Get probabilities
    y_prob = classifier.predict(x)

    # ... and extract for predictions
    y_pred = y_prob
    super_threshold_indices = y_prob > threshold
    y_pred[super_threshold_indices] = 1
    y_pred[np.invert(super_threshold_indices)] = 0

    pp = PrettyPrinter(indent=4)

    metrics['Hamming loss'] = hamming_loss(y, y_pred)
    metrics['Exact match ratio'] = accuracy_score(y, y_pred)
    metrics['Hamming score'] = hamming_score(y, y_pred)
    # metrics['Precision score (micro)'] = precision_score(y, y_pred, average='micro')
    # metrics['Precision score (macro)'] = precision_score(y, y_pred, average='macro')
    metrics['F1 score (micro)'] = f1_score(y, y_pred, average='micro')
    metrics['Label-based accuracy'] = label_based_accuracy(y, y_pred)
    metrics['Jaccard similarity score'] = jaccard_similarity_score(y, y_pred)
    # metrics['F1 score (macro)'] = f1_score(y, y_pred, average='macro')

    """In a multi-class classification setup, micro-average is preferable if you suspect
     there might be class imbalance (i.e you may have many more examples of one class
      than of other classes) -> We know this is not the case."""

    # pp.pprint(metrics)

    return metrics


#############
# Visualize #
#############

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
    colors = sns.color_palette('pastel', n_colors=3)

    # Initialize axes
    fig, subplot_axes = plt.subplots(2, 2,
                                     squeeze=False,
                                     sharex='none',
                                     sharey='none',
                                     figsize=(11, 10),
                                     constrained_layout=True)
    fig.dpi = 150

    # Fill axes
    for col in range(2):

        train_or_val = 'Training' if col == 0 else 'Validation'

        for row in range(2):

            ax = subplot_axes[row][col]

            color_counter = 0
            for label, history in histories.items():

                n_epochs = len(history.history['loss'])

                if row == 0:
                    key = 'acc' if col == 0 else 'val_acc'
                    title = '{} accuracy'.format(train_or_val)
                    y_label = 'Accuracy'
                    y_limit = (0.6, 1)

                else:
                    key = 'loss' if col == 0 else 'val_loss'
                    title = '{} loss'.format(train_or_val)
                    y_label = 'Loss (binary cross entropy)'
                    y_limit = (0, 1)

                # Plot label
                plot_label = "{}".format(label.capitalize()) if row == 0 and col == 0 else ""
                # Plot training & validation accuracy values
                ax.plot(history.history[key], label=plot_label, color=colors[color_counter])

                # Add vertical line to indicate early stopping
                ax.axvline(x=n_epochs - 1, linestyle='--', color=colors[color_counter])

                # Set a title, the correct y-label, and the y-limit
                ax.set_title(title, fontdict={'fontweight': 'semibold', 'family': 'serif'})
                ax.set_ylabel(y_label, fontdict={'family': 'serif'})
                ax.set_ylim(y_limit)

                color_counter += 1

                # ax.set_yscale("log")

            if row == 0 and col == 0:
                ax.legend(loc='best', prop={'family': 'serif'})

            if row == 1: ax.set_xlabel('Epoch', fontdict={'family': 'serif'})
            # ax.set_xlim(0, 100)
            # fig.legend(loc='best', prop={'weight': 'bold', 'family':'serif'})

    # Title
    '''plt.suptitle(
        "Training histories of classifier models (image dim {} - compression {})".format(image_dim, compression_factor),
        fontweight='bold')'''

    # Build model path
    evaluation_label = 'histories_im_dim{}comp{}loss{}'.format(image_dim, compression_factor, loss)
    plot_path = os.path.join(os.pardir, "models", "classifiers", "plots", evaluation_label + ".png")

    # Show 'n tell
    if save: fig.savefig(plot_path, dpi=fig.dpi)
    if plot: plt.show()

    return ax


def plot_model_metrics(evaluations: dict, image_dim: int, compression_factor: int, loss: str, save=True,
                       plot=True) -> plt.Axes:
    """
    Visualize comparison of approaches in terms of chosen metrics.

    :param evaluations: dict containing metrics per approach
    :param image_dim: image dimension
    :param compression_factor: (sic)
    :param save: should we save to disk?
    :param plot: show plot?
    :return: plot axes
    """
    # Pour evaluations into data frame
    evaluations_df = pd.DataFrame(evaluations).reset_index().rename(index=str, columns={'index': 'metric'})
    evaluations_long = pd.melt(evaluations_df, id_vars='metric', var_name='approach', value_name='value')

    # Nested barplot to show evaluation metrics
    sns.set_style('white')
    ax = sns.catplot(x="metric", y="value", data=evaluations_long, kind='bar', legend=None,
                     hue="approach", palette="pastel", height=8, aspect=1.6)
    ax.despine(bottom=True, offset=10, trim=True)
    ax.set_xlabels("", fontdict={'fontweight': 'bold', 'family': 'serif'})
    ax.set_ylabels("")
    ax.set_xticklabels(fontdict={'fontweight': 'bold', 'family': 'serif'})
    plt.legend(loc='best', prop={'weight': 'bold', 'family': 'serif'})

    # Title
    '''plt.suptitle(
        "Evaluation of classifier models (image dim {} - compression {})".format(image_dim, compression_factor),
        fontweight='bold')'''

    # Build model path
    evaluation_label = 'evaluation_metrics_im_dim{}comp{}_loss{}'.format(image_dim, compression_factor, loss)
    model_path = os.path.join(os.pardir, "models", "classifiers", "plots", evaluation_label + ".png")

    # Show 'n tell
    if save: plt.savefig(model_path, dpi=150)
    if plot: plt.show()

    return ax


def plot_classifier_predictions(classifier: Sequential, approach: str, model_name: str, compression_factor: float,
                                loss: str,
                                x: np.ndarray, y_true: np.ndarray,
                                examples=5, random=True, save=True, plot=True):
    """
    Show some images, their true class labels, and the predicted class labels, for a given classifier.

    :param classifier: model used to predict labels
    :param model_name:
    :param compression_factor:
    :param x: images
    :param y_true: true labels
    :param examples: number of images to show
    :param random: random selection of training images, or sequential (i.e. first #examples)
    :param save: save to disk?
    :param plot: show plot?
    :return: SubplotAxes
    """

    log("Plotting classifier predictions.")

    # Set font
    font = {'fontname': 'Times New Roman Bold',
            'fontfamily': 'serif',
            'weight': 'bold'}

    # Set indices
    rnd.seed(818)
    indices = rnd.sample(range(len(x)), examples) if random else [i for i in range(examples)]

    # Take subsets
    x_sample = x[indices]
    y_true_sample = y_true[indices]

    # Make predictions
    y_pred_sample = classifier.predict(x=x_sample)

    # Get image dimension
    image_dim = x.shape[1]

    # Plot parameters
    plot_count = examples * 2
    row_count = 2
    col_count = int(ceil(plot_count / row_count))

    # Initialize axes
    fig, subplot_axes = plt.subplots(row_count,
                                     col_count,
                                     squeeze=True,
                                     sharey='row',
                                     figsize=(18, 9),
                                     constrained_layout=True)

    # Set colors
    colors = sns.color_palette('pastel', n_colors=len(dat.CLASSES))

    # Fill axes
    for i in range(plot_count):

        row = i // col_count
        col = i % col_count

        original_image = x_sample[col]

        ax = subplot_axes[row][col]

        # First row: show original images
        if row == 0:
            labels = [label for got_label, label in zip(y_true_sample[col], dat.CLASSES) if got_label == 1]
            ax.set_title("Image label: {}".format(labels), fontdict=font)
            ax.imshow(original_image)
            ax.axis('off')

        # Second row: show predictions
        else:
            ax.set_title("Predictions", fontdict=font)
            # sns.barplot(y=y_pred_sample[col], hue=colors)
            ax.bar(x=range(len(dat.CLASSES)), height=y_pred_sample[col], color=colors)
            ax.axhline(y=.5, linestyle='--', color='black')
            ax.set_xticks(ticks=range(len(dat.CLASSES)))
            ax.set_xticklabels(dat.CLASSES)
            ax.set_ylim(0, 1)
            # ax.set_aspect(2)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        for i in range(len(x_sample)):
            subplot_axes[1][i].set_ylim(0, 1.0)

    # General make-up
    plt.tight_layout()

    # Full model name for file output
    full_model_name = "predictions_" + model_name + '_im_dim' + str(image_dim) + '-comp' + str(
        int(compression_factor)) + 'loss' + loss

    # Build model path
    model_path = os.path.join(os.pardir, "models", "classifiers", "plots", full_model_name + ".png")

    # Title
    '''plt.suptitle(
        "Predictions for <{}> approach (image dim {} - compression {})".format(approach, image_dim, compression_factor),
        fontweight='bold')'''

    # Show 'n tell
    if save: plt.savefig(model_path, dpi=150)
    if plot: plt.show()

    return fig, subplot_axes, indices


def plot_classifier_prediction_comparison(classifiers: list, model_name: str, compression_factor: float, loss: str,
                                          x: np.ndarray, y_true: np.ndarray,
                                          examples=5, random=True, save=True, plot=True, indices=None):
    """
    Show some images, their true class labels, and the predicted class labels, for a given classifier.

    :param classifier: model used to predict labels
    :param model_name:
    :param compression_factor:
    :param x: images
    :param y_true: true labels
    :param examples: number of images to show
    :param random: random selection of training images, or sequential (i.e. first #examples)
    :param save: save to disk?
    :param plot: show plot?
    :return: SubplotAxes
    """

    log("Plotting classifier prediction comparison.")

    # Set font
    font = {'fontname': 'Times New Roman Bold',
            'fontfamily': 'serif',
            'weight': 'bold'}

    # Set indices
    if not indices:
        rnd.seed(919)
        indices = rnd.sample(range(len(x)), examples) if random else [i for i in range(examples)]
    else:
        indices = indices

    # Take subsets
    x_sample = x[indices]
    y_true_sample = y_true[indices]
    y_pred_sample = []

    # Make predictions
    for classifier in classifiers:
        y_pred_sample.append(classifier.predict(x=x_sample))

    # Get image dimension
    image_dim = x.shape[1]

    # Plot parameters
    row_count = len(classifiers) + 1
    col_count = examples
    plot_count = row_count * col_count
    # Initialize axes
    fig, subplot_axes = plt.subplots(row_count,
                                     col_count,
                                     squeeze=True,
                                     sharey='row',
                                     figsize=(15, 12),
                                     constrained_layout=True)

    # Set colors
    colors = sns.color_palette('pastel', n_colors=len(dat.CLASSES))

    # Fill axes
    for i in range(plot_count):

        row = i // col_count
        col = i % col_count

        original_image = x_sample[col]

        ax = subplot_axes[row][col]

        # First row: show original images
        if row == 0:
            labels = [label for got_label, label in zip(y_true_sample[col], dat.CLASSES) if got_label == 1]
            ax.set_title("Image label: {}".format(labels), fontdict=font)
            ax.imshow(original_image)
            ax.axis('off')

        # Second, third, ... row: show predictions
        else:
            ax.set_title("Predictions", fontdict=font)
            # sns.barplot(y=y_pred_sample[col], hue=colors)
            ax.bar(x=range(len(dat.CLASSES)), height=y_pred_sample[row - 1][col], color=colors)
            ax.axhline(y=.5, linestyle='--', color='black')
            ax.set_xticks(ticks=range(len(dat.CLASSES)))
            ax.set_xticklabels(dat.CLASSES)
            ax.set_ylim(0, 1)
            # ax.set_aspect(2)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        for i in range(len(x_sample)):
            subplot_axes[1][i].set_ylim(0, 1.0)

    # General make-up
    plt.tight_layout()

    # Full model name for file output
    full_model_name = "prediction_comparison" + model_name + '_im_dim' + str(image_dim) + '-comp' + str(
        int(compression_factor)) + 'loss' + loss

    # Build model path
    model_path = os.path.join(os.pardir, "models", "classifiers", "plots", full_model_name + ".png")

    # Title
    '''plt.suptitle(
        "Predictions for <{}> approach (image dim {} - compression {})".format(approach, image_dim, compression_factor),
        fontweight='bold')'''

    # Show 'n tell
    if save: plt.savefig(model_path, dpi=150)
    if plot: plt.show()

    return fig, subplot_axes


########
# MAIN #
########

if __name__ == "__main__":

    # Let's go
    log("CLASSIFIERS", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('classifiers')

    # Dashboard
    image_dim = 96
    compression_factor = 24
    loss = 'binary_crossentropy'

    # Set remaining parameters
    autoencoder_name = 'conv_auto'
    classifier_name = 'conv_class'
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    # Plot parameters
    font = {'family': 'serif',
            'weight': 'bold'}
    matplotlib.rc('font', **font)
    rnd.seed(616)

    # Train or just load?
    train = False

    # Get the data
    data, _ = dat.pipeline(image_dim=image_dim)
    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']
    x_te, y_te = data['x_test'], data['y_test']

    # Getting number of class objects per dataset
    ''' 
    print(np.unique([dat.CLASSES[i] for i in [np.where(r == 1)[0][0] for r in y_tr]], return_counts=True))
    print(np.unique([dat.CLASSES[i] for i in [np.where(r == 1)[0][0] for r in y_va]], return_counts=True))
    print(np.unique([dat.CLASSES[i] for i in [np.where(r == 1)[0][0] for r in y_te]], return_counts=True))
    '''

    # Store classifiers and their histories
    classifiers, histories, evaluations, comparison = {}, {}, {}, []

    # Model parameters to loop over
    parameter_combinations = [(False, False),
                              (False, True),
                              (True, True)]

    # Build classifiers
    for index, (from_scratch, all_trainable) in enumerate(parameter_combinations):

        # Detail the step we're taking
        approach = "{}from scratch, {}all layers trainable".format(("" if from_scratch else "not "),
                                                                   ("" if all_trainable else "not "))
        log("Classifier {}".format(index + 1), title=True)
        log("Approach: {}.".format(approach), lvl=1)

        # Full model name for file output
        full_autoencoder_name = "{}_im_dim{}-comp{}".format(autoencoder_name, image_dim, compression_factor)
        full_classifier_name = "{}_im_dim{}-comp{}_full{}_scratch{}_loss{}".format(classifier_name, image_dim,
                                                                                   compression_factor,
                                                                                   all_trainable, from_scratch,
                                                                                   loss)

        # Build paths
        log("Building paths.", lvl=3)
        autoencoder_path = os.path.join(os.pardir, "models", "autoencoders", full_autoencoder_name + ".h5")
        classifier_path = os.path.join(os.pardir, "models", "classifiers", full_classifier_name + ".h5")
        history_path = os.path.join(os.pardir, "models", "classifiers", "history",
                                    full_classifier_name + "_history.npy")

        # Load classifier and history if they exists...
        try:
            classifiers[full_classifier_name] = load_model(filepath=classifier_path)
            log("Found model \'", full_classifier_name, "\' locally.", lvl=3)
            histories[approach] = np.load(file=history_path).item()
            log("Found model \'", full_classifier_name, "\' history locally.", lvl=3)

        # ... build it if it doesn't
        except:
            log("Failed to find model \'{}\' - building.".format(full_classifier_name), lvl=2)
            classifiers[full_classifier_name], histories[approach] = build_classifier(image_dim=image_dim,
                                                                                      compression_factor=compression_factor,
                                                                                      x_tr=x_tr, y_tr=y_tr, x_va=x_va,
                                                                                      y_va=y_va,
                                                                                      loss=loss,
                                                                                      from_scratch=from_scratch,
                                                                                      all_trainable=all_trainable,
                                                                                      epochs=300, patience=0)

            # Save history
            log("Saving history.", lvl=3)
            np.save(file=history_path, arr=histories[approach])

        comparison.append(classifiers[full_classifier_name])

        # Evaluate classifier
        evaluations[approach] = evaluate_classifier(classifiers[full_classifier_name], x=x_te, y=y_te, threshold=.5)

        # Show some predictions
        _, _, indices = plot_classifier_predictions(classifier=classifiers[full_classifier_name], approach=approach,
                                                    model_name=full_classifier_name,
                                                    compression_factor=compression_factor,
                                                    loss=loss,
                                                    x=x_te, y_true=y_te, examples=5, random=True,
                                                    save=True, plot=True)

    # Plot model histories
    plot_model_histories(histories=histories, save=True, plot=True, image_dim=image_dim,
                         compression_factor=compression_factor, loss=loss)

    # Plot model metrics
    plot_model_metrics(evaluations=evaluations, save=True, plot=True, image_dim=image_dim,
                       compression_factor=compression_factor, loss=loss)

    # Plot prediction comparison
    plot_classifier_prediction_comparison(classifiers=comparison,
                                          model_name='class_comparison',
                                          compression_factor=compression_factor,
                                          loss=loss, x=x_te, y_true=y_te, examples=5,
                                          random=True, save=True, plot=True,
                                          indices=indices)
