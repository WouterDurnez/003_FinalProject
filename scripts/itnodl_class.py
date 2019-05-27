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

import itnodl_data as dat
import itnodl_help as hlp
import tensorflow as tf
from itnodl_auto import build_autoencoder
from itnodl_help import log, make_folders
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Activation, Dropout

if __name__ == "__main__":
    # Let's go
    log("CLASSIFIERS", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Set model parameters
    model_name = 'conv_auto'
    classifier_name = 'conv_class'
    image_dim = 16
    image_size = image_dim ** 2 * 3
    compression_factor = 32
    encoding_dim = image_size // compression_factor

    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))
    full_classifier_name = classifier_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))

    # Build paths
    model_path = os.path.join(os.pardir, "models", full_model_name + ".h5")
    classifier_path = os.path.join(os.pardir, "models", full_classifier_name + ".h5")
    history_path = os.path.join(os.pardir, "models", "history", full_model_name + "_history.npy")

    # Get the data
    (x_tr, y_tr), (x_va, y_va) = dat.pipeline(image_dim=image_dim)

    # Build encoder
    _, encoder, _ = build_autoencoder(model_name='conv_auto',
                                      convolutional=True,
                                      train=False,
                                      x_tr=x_tr, x_va=x_va,
                                      encoding_dim=encoding_dim,
                                      epochs=75,
                                      patience=10)

    # Freeze, dirtbag
    '''encoder.get_layer(index=1).trainable = False
    encoder.get_layer(index=2).trainable = False
    encoder.get_layer(index=4).trainable = False
    encoder.get_layer(index=5).trainable = False'''

    # Adapting into classifier
    classifier = Sequential()
    classifier.add(encoder)
    classifier.add(Flatten())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(5, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    encoder.summary()
    classifier.summary()

    # Train the network
    # Training parameters
    patience = 25
    verbose = 1

    es = EarlyStopping(monitor='val_acc', patience=patience, verbose=verbose)
    mc = ModelCheckpoint(filepath=classifier_path, monitor='val_acc', verbose=verbose, save_best_only=True)
    tb = TensorBoard(log_dir='/tmp/' + classifier_name + '_im' + str(image_dim) + 'comp' + str(int(compression_factor)))

    # Train model
    classifier.fit(x_tr, y_tr,
                   epochs=200,
                   batch_size=32,
                   verbose=verbose,
                   validation_data=(x_va, y_va),
                   callbacks=[es, mc, tb])

    # Save model and history
    y_va_pre = classifier.predict(x=x_va)
    # encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='encoder').output)

# Freeze
