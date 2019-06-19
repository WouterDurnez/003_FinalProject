#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
TASK 3 - EXTRA CLASSIFIER -- AUTOENCODER/CLASSIFIER COMBINATION

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, History
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.utils import plot_model

import itnodl_data as dat
import itnodl_help as hlp
from itnodl_help import log, set_up_model_directory


######################################
# Autoencoder/classifier combination #
######################################

def joint_model(model_name: str, image_dim: int, compression_factor: int, verbose=1) -> (Model, History):
    """
    Build and train a joint model: autoencoder plus classification training.

    :param model_name: name for file output
    :param image_dim: dimension of input image
    :param compression_factor: (sic)
    :param verbose: verbosity during training.

    :return: joint model, plus its history
    """

    # Full model name for file output
    full_model_name = "{}_im{}comp{}".format(model_name, image_dim, compression_factor)

    # Build model paths
    model_path = os.path.join(os.pardir, "models", "classifiers", full_model_name + ".h5")
    architecture_path = os.path.join(os.pardir, "models", "classifiers", "architecture",
                                     full_model_name + "_architecture.png")
    history_path = os.path.join(os.pardir, "models", "classifiers", "history", full_model_name + "_history.npy")

    # Try loading the model, ...
    try:

        joint = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path)
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except:

        # Model parameters
        input_shape = (image_dim, image_dim, 3)
        image_size = np.prod(input_shape)
        encoding_dim = image_size // compression_factor

        # Build model base (encoder)
        image_input = Input(shape=input_shape)
        conv_1 = Conv2D(image_dim, (3, 3), padding='same', activation='relu',
                        kernel_initializer='random_uniform', bias_initializer='zeros')(image_input)
        batch_1 = BatchNormalization()(conv_1)
        max_1 = MaxPooling2D((2, 2), padding='same')(batch_1)
        conv_2 = Conv2D(image_dim // (2 * compression_factor), (3, 3), padding='same', activation='relu',
                        kernel_initializer='random_uniform', bias_initializer='zeros')(max_1)
        batch_2 = BatchNormalization()(conv_2)
        encoded = MaxPooling2D((2, 2), padding='same')(batch_2)

        # Build autoencoder
        conv_3 = Conv2D(image_dim // (2 * compression_factor), (3, 3), padding='same', activation='relu',
                        kernel_initializer='random_uniform', bias_initializer='zeros')(encoded)
        batch_3 = BatchNormalization()(conv_3)
        up_1 = UpSampling2D((2, 2))(batch_3)
        conv_4 = Conv2D(image_dim, (3, 3), padding='same', activation='relu',
                        kernel_initializer='random_uniform', bias_initializer='zeros')(up_1)
        batch_4 = BatchNormalization()(conv_4)
        up_2 = UpSampling2D((2, 2))(batch_4)
        autoencoder = Conv2D(3, (10, 10), padding='same', activation='sigmoid',
                             kernel_initializer='random_uniform', bias_initializer='zeros', name='autoencoder')(up_2)

        # Build classifier
        flatten = Flatten()(encoded)
        dense_1 = Dense(encoding_dim, activation='relu')(flatten)
        drop_1 = Dropout(.5)(dense_1)
        classifier = Dense(5, activation='sigmoid', name='classifier')(drop_1)

        # Build joint model
        joint = Model(inputs=image_input, outputs=[autoencoder, classifier])

        # Save model architecture visualization
        plot_model(joint, to_file=architecture_path)

        joint.compile(loss={'classifier': 'binary_crossentropy',
                            'autoencoder': 'mean_squared_error'},
                      optimizer='adam',
                      metrics={'classifier': 'accuracy'})

        # Callbacks
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
        mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        tb = TensorBoard(log_dir="/tmp/{}_im{}comp{}".format(model_name, image_dim, compression_factor))

        history = joint.fit(x=x_tr, y={'classifier': y_tr, 'autoencoder': x_tr},
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_te, {'classifier': y_te, 'autoencoder': x_te}),
                            verbose=1,
                            callbacks=[es, mc, tb])

        np.save(file=history_path, arr=history)

    return joint, history


if __name__ == "__main__":
    # Let's go
    log("JOINT MODEL - AUTOENCODER/CLASSIFIER", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('classifiers')

    # Dashboard
    image_dim = 96
    compression_factor = 24
    epochs = 300
    patience = 50
    batch_size = 128

    # Get the data
    data, _ = dat.pipeline(image_dim=image_dim)
    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']
    x_te, y_te = data['x_test'], data['y_test']

    # Build model
    model, history = joint_model(model_name='comb_model', image_dim=image_dim, compression_factor=compression_factor)
