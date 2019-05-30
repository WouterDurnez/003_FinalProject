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
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, History
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import itnodl_help as hlp
from itnodl_auto import build_autoencoder
from itnodl_data import pipeline
from itnodl_help import log, set_up_model_directory


def build_classifier(image_dim: int, compression_factor: int,
                     x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
                     all_trainable=False, epochs=200, patience=25) -> (Sequential, History):
    # Build encoder
    _, encoder, _ = build_autoencoder(model_name='conv_auto',
                                      convolutional=True,
                                      train=False,
                                      x_tr=x_tr, x_va=x_va,
                                      compression_factor=compression_factor)
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    # Freeze, dirtbag (or not, I'm not dirty Harry)
    conv_layers = [1, 2, 4, 5]
    for i in conv_layers:
        encoder.get_layer(index=i).trainable = all_trainable

    # Extending into classifier

    input_shape = (image_dim, image_dim, 3)

    classifier = Sequential()

    # classifier.add(encoder)
    classifier.add(Conv2D(2 * encoding_dim, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D((2, 2), padding='same'))

    classifier.add(Conv2D(encoding_dim, (3, 3), padding='same', activation='relu',
                          kernel_initializer='random_uniform', bias_initializer='zeros'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D((2, 2), padding='same', name='encoder'))

    classifier.add(Flatten())
    classifier.add(Dense(encoding_dim))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(5, activation='sigmoid'))  # <-- multilabel (for multiclass: softmax)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_tr)

    # Training parameters
    verbose = 2

    # Callbacks
    if patience == 0:
        patience = epochs
    es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
    mc = ModelCheckpoint(filepath=classifier_path, monitor='val_loss', verbose=verbose, save_best_only=True)
    tb = TensorBoard(log_dir='/tmp/' + classifier_name + '_im' + str(image_dim) + 'comp' + str(int(compression_factor)))

    # Train model
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
    np.save(file=history_path, arr=history)

    return classifier, history


if __name__ == "__main__":
    # TODO: add data generator

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

    # Full model name for file output
    full_autoencoder_name = "{}_im_dim{}-comp{}".format(autoencoder_name, image_dim, compression_factor)
    full_classifier_name = "{}_im_dim{}-comp{}".format(classifier_name, image_dim, compression_factor)

    # Build paths
    autoencoder_path = os.path.join(os.pardir, "models", "autoencoders", full_autoencoder_name + ".h5")
    classifier_path = os.path.join(os.pardir, "models", "classifiers", full_classifier_name + ".h5")
    history_path = os.path.join(os.pardir, "models", "classifiers", "history", full_classifier_name + "_history.npy")

    # Get the data
    (x_tr, y_tr), (x_va, y_va) = pipeline(image_dim=image_dim)

    # Build classifier
    classifier, history = build_classifier(image_dim=image_dim, compression_factor=compression_factor,
                                           x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, all_trainable=True, patience=20)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
