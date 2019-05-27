#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
DATA OPERATIONS

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os
from math import sqrt
from xml.etree import ElementTree as ET

import itnodl_help as hlp
import numpy as np
from itnodl_help import time_it, log, make_folders
from skimage import io
from skimage.transform import resize


@time_it
def build_classification_dataset(list_of_files: list, image_dim: int, filter: list, voc_root_folder: str):
    """
    Build training or validation set.

    :param list_of_files: list of filenames to build trainset with
    :param image_dim: taken from upper level function
    :param voc_root_folder: taken from upper level function
    :return: tuple with x np.ndarray of shape (n_images, image_dim, image_dim, 3) and  y np.ndarray of shape (n_images, n_classes)
    """

    temp, train_labels = [], []

    # Loop over files
    for f_cf in list_of_files:
        # Open file
        with open(f_cf) as file:
            # Split file up in lines
            lines = file.read().splitlines()

            # Add list of lines (image names) to temporary list (minus trailing '-1')
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])

            # Get label ids and add them to the list
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]

            train_labels.append(len(temp[-1]) * [label_id])

    # Concatenate item names
    train_filter = [item for l in temp for item in l]

    # Get image filenames
    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]

    # Get 'raw' image data and add to list
    x = np.array([resize(io.imread(img_f), (image_dim, image_dim, 3)) for img_f in image_filenames],dtype='float32')

    # Change y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y


def pipeline(image_dim=214,
             filter=['aeroplane', 'car', 'chair', 'dog', 'bird'],
             voc_root_folder="../data/VOCdevkit/") -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """
    Pipeline to extract training and validation datasets from full dataset.

    :param image_dim: images will be resized to image_dim x image_dim pixels
    :param filter: classes to take into account (out of total of 20)
    :param voc_root_folder: root folder for voc dataset
    :return:
    """

    log("Preparing data.", lvl=1)

    # Make folders, should they not exist yet
    make_folders("data", "scripts")

    # Try to load data sets locally
    make_folders(os.path.join('data', 'image_dim_'+str(image_dim)))
    x_train_path = os.path.join(os.pardir, 'data', 'image_dim_'+str(image_dim), "x_train_" + str(image_dim) + ".npy")
    y_train_path = os.path.join(os.pardir, 'data', 'image_dim_'+str(image_dim), "y_train_" + str(image_dim) + ".npy")
    x_val_path = os.path.join(os.pardir, 'data', 'image_dim_'+str(image_dim), "x_val_" + str(image_dim) + ".npy")
    y_val_path = os.path.join(os.pardir, 'data', 'image_dim_'+str(image_dim), "y_val_" + str(image_dim) + ".npy")

    try:

        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
        log("Data loaded.", lvl=3)
        return (x_train, y_train), (x_val, y_val)

    except (NameError, FileNotFoundError):

        log("Failed to load one of the {}-sized datasets. Rebuilding.".format(image_dim), lvl=1)

    # Build list of filtered filenames #
    log("Building list of filtered filenames.", lvl=2)

    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []

    # Go over all annotation XML files
    for a_f in annotation_files:

        # Parse each file
        tree = ET.parse(os.path.join(annotation_folder, a_f))

        # Check for presence of each of the filter classes
        check = [tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]

        # If any of them are present, add it to the list (and disregard the extension)
        if np.any(check):
            filtered_filenames.append(a_f[:-4])

    # Build (x,y) for TRAIN/VAL (classification)
    log("Building filtered dataset.", lvl=2)

    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)

    # Only take training and validation images that pass filter
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if
                   filt in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if
                 filt in c_f and '_val.txt' in c_f]

    x_train, y_train = build_classification_dataset(list_of_files=train_files, image_dim=image_dim, filter=filter,
                                                    voc_root_folder=voc_root_folder)
    log("Extracted {} training images from {} classes.".format(x_train.shape[0], y_train.shape[1]), lvl=3)
    x_val, y_val = build_classification_dataset(list_of_files=val_files, image_dim=image_dim, filter=filter,
                                                voc_root_folder=voc_root_folder)
    log("Extracted {} validation images from {} classes.".format(x_val.shape[0], y_train.shape[1]), lvl=3)

    # Save locally, because this takes a while
    log("Storing data.", lvl=3)
    np.save(x_train_path, x_train)
    np.save(y_train_path, y_train)
    np.save(x_val_path, x_val)
    np.save(y_val_path, y_val)

    return (x_train, y_train), (x_val, y_val)


def squash(x: np.ndarray) -> np.ndarray:
    """
    Flatten input matrix (n images of size image_dim x image_dim x 3).

    :param x: input matrix (
    :return: flattened input matrix
    """

    if len(x.shape) > 2:
        image_dim = x.shape[1]
        image_size = image_dim ** 2 * 3
        return x.reshape((len(x), image_size))
    else:
        log("Failed to squash x - already flattened.", lvl=3)
        return x


def lift(x: np.ndarray) -> np.ndarray:
    """
    Expand input matrix (n image vectors of size image_dim x image_dim x 3).

    :param x: input matrix
    :return: expanded input matrix
    """

    if not len(x.shape) > 2:
        image_size = x.shape[1]
        image_dim = round(sqrt(image_size/3))
        return x.reshape((len(x), image_dim, image_dim, 3))
    else:
        log("Failed to lift x - already expanded.", lvl=3)
        return x


if __name__ == '__main__':

    hlp.LOG_LEVEL = 3

    (a, b), (c, d) = pipeline(image_dim=12)
