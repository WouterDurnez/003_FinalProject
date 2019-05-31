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
import numpy as np

import itnodl_help as hlp
from itnodl_help import log, set_up_model_directory, make_folders
from itnodl_data import pipeline
import tensorflow as tf

if __name__ == "__main__":
    # Let's go
    log("CLASSIFIERS", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Check folders
    set_up_model_directory('segmentation')

    # Set model parameters
    autoencoder_name = 'conv_auto'
    segmentation_name = 'conv_segm'
    image_dim = 32
    image_size = image_dim ** 2 * 3
    compression_factor = 32
    encoding_dim = image_size // compression_factor

    # Get the data
    (x_tr, y_tr), (x_va, y_va) = pipeline(image_dim=image_dim)
