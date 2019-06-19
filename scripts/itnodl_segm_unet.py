#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
TASK 4 - EXTRA SEGMENTATION NETWORK -- U-NET

Coded by Wouter Durnez
Adapted from: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, History
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

import itnodl_data as dat
import itnodl_help as hlp
import itnodl_segm as seg
from itnodl_data import threshold
from itnodl_help import log, set_up_model_directory


####################################
# U-Net based segmentation network #
####################################

def build_unet_segmentation_network(model_name: str,
                                    x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray,
                                    optimizer='adam', loss='mean_squared_error', epochs=200, patience=25,
                                    verbose=1) -> (Model, History):
    """
    Build and train U-net inspired segmentation network.

    :param x_tr: training images
    :param y_tr: training segmentations
    :param x_va: validation images
    :param y_va: validation segmentations
    :param optimizer: (sic)
    :param loss: (sic)
    :param epochs: (sic)
    :param patience: (sic)
    :param verbose: (sic)

    :return: U-net model and its training history
    """

    # Set parameters
    image_dim = x_tr.shape[1]
    image_size = image_dim ** 2 * 3
    input_shape = (image_dim, image_dim, 3)

    # Build model path
    model_name = "{}_im{}".format(model_name, image_dim)
    model_path = os.path.join(os.pardir, "models", "segmentation", model_name + ".h5")
    architecture_path = os.path.join(os.pardir, "models", "segmentation", "architecture",
                                     model_name + "_architecture.png")
    history_path = os.path.join(os.pardir, "models", "segmentation", "history", model_name + "_history.npy")

    # Try loading the model, ...
    try:

        unet = load_model(model_path)
        log("Found model \'", model_name, "\' locally.", lvl=3)
        history = np.load(file=history_path).tolist()
        log("Found model \'", model_name, "\' history locally.", lvl=3)

    # ... otherwise, create it
    except Exception as e:

        log("Exception:", e, lvl=3)

        # Build U-Net model
        image_input = Input(shape=input_shape)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(image_input)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        unet = Model(inputs=[image_input], outputs=[outputs])
        unet.compile(optimizer=optimizer, loss=loss)
        unet.summary()

        plot_model(unet, to_file=architecture_path, show_shapes=True)

        # Callbacks
        if patience == 0:
            patience = epochs
        es = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
        mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=verbose, save_best_only=True)
        tb = TensorBoard(log_dir="/tmp/segm_unet_im{}".format(image_dim))

        # Data augmentation to get the most out of our images
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(x_tr)

        # Train model using data augmentation
        history = unet.fit_generator(datagen.flow(x_tr, y_tr, batch_size=128),
                                     epochs=epochs,
                                     steps_per_epoch=x_tr.shape[0] // 128,
                                     verbose=verbose,
                                     validation_data=(x_va, y_va),
                                     callbacks=[es, mc, tb])

        # Save model and history
        np.save(file=history_path, arr=history)
    plot_model(unet, to_file=architecture_path, show_shapes=True)

    return unet, history


if __name__ == "__main__":

    # Let's go
    log("U-NET SEGMENTATION NETWORK", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('segmentation')

    # Dashboard
    image_dim = 96
    compression_factor = 24
    epochs = 500
    patience = 0
    loss = 'mean_squared_error'

    # Set remaining parameters
    segmentation_name = 'conv_segm_unet'
    image_size = image_dim ** 2 * 3
    encoding_dim = image_size // compression_factor

    # Get the data
    _, data = dat.pipeline(image_dim=image_dim, class_data=True, segm_data=True, mono=True)

    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']
    x_te, y_te, y_te_src = data['x_test'], data['y_test'], data['y_test_source']

    # Train segmentation network
    unet, history = build_unet_segmentation_network(model_name=segmentation_name, x_tr=x_tr, y_tr=y_tr, x_va=x_va,
                                                    y_va=y_va,
                                                    epochs=epochs, optimizer='adam', loss=loss, patience=patience)

    # Evaluation
    unet_eval = []
    unet_dice = []

    y_tr_pred = unet.predict(x_tr)
    y_va_pred = unet.predict(x_va)
    y_te_pred = unet.predict(x_te)

    for x, y, y_pred in zip([x_tr, x_va, x_te], [y_tr, y_va, y_te], [y_tr_pred, y_va_pred, y_te_pred]):
        unet_eval.append(unet.evaluate(x, y))
        # unet_dice.append(dice_loss(y, unet.predict(x)))
        unet_dice.append(1 - seg.dice_coef(y, y_pred))

    print("MSE\t", unet_eval)
    print("Dice\t", unet_dice)

    # Visualize results
    y_te_res = x_te.copy()
    for color_channel in range(3):
        y_te_res[:, :, :, color_channel] = np.multiply(x_te[:, :, :, color_channel],
                                                       threshold(y_te_pred, threshold=.5, mono=False)[:, :, :, 0])

    seg.plot_segmentation_results(image_list=[x_tr, y_tr, y_tr_pred, threshold(y_tr_pred, threshold=.5, mono=False)],
                                  labels=["Original", "Target", "Predicted", "Threshold"],
                                  examples=5, random=True, save=True, plot=True, name="train", seed=87)
    #  indices=[49, 109, 147]

    seg.plot_segmentation_results(image_list=[x_va, y_va, y_va_pred, threshold(y_va_pred, threshold=.5, mono=False)],
                                  labels=["Original", "Target", "Predicted", "Threshold"],
                                  examples=5, random=True, save=True, plot=True, name="val")

    seg.plot_segmentation_results(
        image_list=[x_te, y_te, y_te_pred, threshold(y_te_pred, threshold=.5, mono=False), y_te_res],
        labels=["Original", "Target", "Predicted", "Threshold", "Masked"],
        examples=5, random=True, save=True, plot=True, name="test")

    # Visualize histories
    seg.plot_model_history(history=history, image_dim=image_dim, save=True, plot=True, model_type="segmentation")
