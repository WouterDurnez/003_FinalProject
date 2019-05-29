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
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.models import Sequential
from keras.utils import plot_model

import itnodl_help as hlp
from itnodl_auto import build_autoencoder
from itnodl_data import pipeline
from itnodl_help import log

if __name__ == "__main__":

    # TODO: add data generator

    # Let's go
    log("CLASSIFIERS", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Set model parameters
    model_name = 'conv_auto'
    classifier_name = 'conv_class'
    image_dim = 32
    image_size = image_dim ** 2 * 3
    compression_factor = 32
    encoding_dim = image_size // compression_factor

    # Full model name for file output
    full_model_name = model_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))
    full_classifier_name = classifier_name + '_im_dim' + str(image_dim) + '-comp' + str(int(compression_factor))

    # Build paths
    model_path = os.path.join(os.pardir, "models", full_model_name + ".h5")
    classifier_path = os.path.join(os.pardir, "models", full_classifier_name + ".h5")
    history_path = os.path.join(os.pardir, "models", "history", full_classifier_name + "_history.npy")

    # Get the data
    (x_tr, y_tr), (x_va, y_va) = pipeline(image_dim=image_dim)

    # Build encoder
    _, encoder, _ = build_autoencoder(model_name='conv_auto',
                                      convolutional=True,
                                      train=False,
                                      x_tr=x_tr, x_va=x_va,
                                      encoding_dim=encoding_dim)

    # Freeze, dirtbag
    train_all = True
    conv_layers = [1, 2, 4, 5]
    for i in conv_layers:
        encoder.get_layer(index=i).trainable = train_all

    # Extending into classifier
    classifier = Sequential()
    classifier.add(encoder)
    classifier.add(Flatten())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(5, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    encoder.summary()
    classifier.summary()

    plot_model(classifier)
    # Train the network
    # Training parameters
    patience = 15
    verbose = 2

    es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
    mc = ModelCheckpoint(filepath=classifier_path, monitor='val_loss', verbose=verbose, save_best_only=True)
    tb = TensorBoard(log_dir='/tmp/' + classifier_name + '_im' + str(image_dim) + 'comp' + str(int(compression_factor)))

    # Train model
    history = classifier.fit(x_tr, y_tr,
                             epochs=200,
                             batch_size=32,
                             verbose=verbose,
                             validation_data=(x_va, y_va),
                             callbacks=[es, mc, tb])

    # Save model and history
    classifier.save(classifier_path)
    np.save(file=history_path, arr=history)

    y_va_pre = classifier.predict(x=x_va)
    y_tr_pre = classifier.predict(x=y_tr)
    # encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='encoder').output)

    count_true = 0
    for i in range(len(y_tr)):
        equal = np.array_equal(y_tr[i], np.round(y_tr_pre[i]).astype('int'))
        print("{} - {} -> {}".format(y_tr[i],
                                     np.round(y_tr_pre[i], 3),
                                     equal)
              )
        if equal:
            count_true += 1
    print("Accuracy: ", count_true * 100 / len(y_tr))

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
