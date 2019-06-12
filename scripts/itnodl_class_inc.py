import os

import tensorflow as tf
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import itnodl_data as dat
import itnodl_help as hlp
from itnodl_help import log, set_up_model_directory

if __name__ == "__main__":

    # Let's go
    log("INCEPTION CLASSIFIER", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('classifiers')

    # Dashboard
    image_dim = 96  # <-- image size must at least be 75
    epochs = 200
    patience = 20
    loss = 'binary_crossentropy'
    verbose = 1

    # Set remaining parameters
    classifier_name = 'conv_inc'
    classifier_path = os.path.join(os.pardir, "models", "classifiers", classifier_name + ".h5")
    image_size = image_dim ** 2 * 3

    # Get the data
    data, _ = dat.pipeline(image_dim=image_dim)
    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']
    x_te, y_te = data['x_test'], data['y_test']

    # Get inception architecture to use as base
    log('Loading INCEPTION base.')
    inc_base = InceptionResNetV2(weights='imagenet', include_top=False,
                                 input_shape=(96, 96, 3))

    # Add classifiers layers on top
    log('Adding classification layers.')
    classifier = Sequential()
    classifier.add(inc_base)
    classifier.add(Flatten())
    classifier.add(Dense(512))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(5, activation='sigmoid'))  # <-- multilabel (for multiclass: softmax)

    # Freeze inception layers
    log('Freezing INCEPTION base.')
    print('# trainable weights before freezing: {}'.format(len(classifier.trainable_weights)))
    inc_base.trainable = False
    print('# trainable weights after freezing: {}'.format(len(classifier.trainable_weights)))

    # Compile model
    classifier.compile(optimizer='Adam', loss=loss, metrics=['acc'])

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
    tb = TensorBoard(log_dir='/tmp/{}_class_im{}'.format(classifier_name, image_dim))

    # Print model info
    log("Network parameters: image dimension {}, image size {}, parameters {}.".
        format(image_dim, image_size, classifier.count_params()), lvl=3)

    # Train model using data augmentation
    history = classifier.fit_generator(datagen.flow(x_tr, y_tr, batch_size=32),
                                       epochs=epochs,
                                       steps_per_epoch=x_tr.shape[0] // 32,
                                       verbose=verbose,
                                       validation_data=(x_va, y_va),
                                       callbacks=[es, mc, tb])
