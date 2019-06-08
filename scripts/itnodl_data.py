#  ___        _   _
# |_ _|_ _   | |_| |_  ___   _ _  __ _ _ __  ___
#  | || ' \  |  _| ' \/ -_) | ' \/ _` | '  \/ -_)
# |___|_||_|  \__|_||_\___| |_||_\__,_|_|_|_\___|    _
#  ___ / _|  __| |___ ___ _ __  | |___ __ _ _ _ _ _ (_)_ _  __ _
# / _ \  _| / _` / -_) -_) '_ \ | / -_) _` | '_| ' \| | ' \/ _` |
# \___/_|   \__,_\___\___| .__/ |_\___\__,_|_| |_||_|_|_||_\__, |
#                        |_|                               |___/

"""
TASK 1 - *Arnold voice* WE AVE TOE GET TO THE DATAA

Coded by Wouter Durnez
-- Wouter.Durnez@UGent.be
-- Wouter.Durnez@student.kuleuven.be
"""

import os
import random as rnd
from math import sqrt
from xml.etree import ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import io
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize

import itnodl_help as hlp
from itnodl_help import time_it, log, make_folders

# GLOBAL VARIABLES #
####################

# Set class labels
CLASSES = ('aeroplane', 'car', 'chair', 'dog', 'bird')

# Set random seed
rnd.seed(616)


##################
# Build datasets #
##################

@time_it
def build_classification_dataset(list_of_files: list, image_dim: int, filter: list, voc_root_folder: str) -> (
        np.ndarray, np.ndarray, list):
    """
    Build training, validation, or test set.

    :param list_of_files: list of filenames to build dataset with
    :param image_dim: taken from upper level function
    :param voc_root_folder: taken from upper level function
    :return: tuple with x np.ndarray of shape (n_images, image_dim, image_dim, 3) and  y np.ndarray of shape (n_images, n_classes)
    """

    temp, labels = [], []

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
            label_check = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf]

            labels.append(len(temp[-1]) * [label_id])

    # Concatenate item names
    image_filter = [item for l in temp for item in l]

    # Get image filenames
    image_folder = os.path.join(voc_root_folder, "VOC2009", "JPEGImages")
    image_filenames = [os.path.join(image_folder, file) for f in image_filter for file in os.listdir(image_folder) if
                       f in file]

    # Get 'raw' image data and add to list
    x = np.array([resize(io.imread(img_f), (image_dim, image_dim, 3)) for img_f in image_filenames], dtype='float32')

    # Change y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in image_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y, image_filter


@time_it
def build_segmentation_dataset(train_list: list, val_list: list, image_dim: int,
                               voc_root_folder: str, mono=True) -> dict:
    """
    Build dataset for segmentation network.
    :param train_list: list of files in training set
    :param val_list: list of files in validation set
    :param image_dim: image dimension
    :param test_indices: indices to serve as test, others remain validation
    :param voc_root_folder: dataset root folder
    :return: dictionary containing original and segmented images (train, test & val)
    """

    # Set paths
    segm_folder = os.path.join(voc_root_folder, "VOC2009", "SegmentationObject")
    image_folder = os.path.join(voc_root_folder, "VOC2009", "JPEGImages")
    segm_files = os.listdir(segm_folder)

    # Get images that have segmented counterpart
    train_segm = [s_f[:-4] for s_f in segm_files for ti in train_list if ti in s_f]
    val_segm = [s_f[:-4] for s_f in segm_files for vi in val_list if vi in s_f]

    # Initialize arrays
    x_train = np.ndarray(shape=(len(train_segm), image_dim, image_dim, 3))
    y_train = np.ndarray(shape=(len(train_segm), image_dim, image_dim, 3))
    x_val = np.ndarray(shape=(len(val_segm), image_dim, image_dim, 3))
    y_val = np.ndarray(shape=(len(val_segm), image_dim, image_dim, 3))

    # Build training images and segmented images
    for idx, image_name in enumerate(train_segm):
        x_train[idx] = resize(io.imread(os.path.join(image_folder, image_name + ".jpg")), (image_dim, image_dim, 3))
        y_train[idx] = resize(cv2.imread(os.path.join(segm_folder, image_name + ".png")), (image_dim, image_dim, 3))

    for idx, image_name in enumerate(val_segm):
        x_val[idx] = resize(io.imread(os.path.join(image_folder, image_name + ".jpg")), (image_dim, image_dim, 3))
        y_val[idx] = resize(cv2.imread(os.path.join(segm_folder, image_name + ".png")), (image_dim, image_dim, 3))

    # Keep random sample as test
    test_indices = rnd.sample(range(x_val.shape[0]), k=len(x_val) // 2)

    x_test = x_val[test_indices]
    y_test = y_val[test_indices]

    x_val = x_val[np.invert(test_indices)]
    y_val = y_val[np.invert(test_indices)]

    # Merge in dictionary
    data_segm = {
        'x_train': x_train,
        'y_train_source': y_train,
        'y_train': threshold(y_train, mono=mono),
        'x_val': x_val,
        'y_val_source': y_val,
        'y_val': threshold(y_val, mono=mono),
        'x_test': x_test,
        'y_test_source': y_test,
        'y_test': threshold(y_test, mono=mono),
    }

    return data_segm


def get_filtered_filenames(voc_root_folder="../data/VOCdevkit/", filter=CLASSES) -> list:
    """
    Get a list of filtered filenames (no extension), based on a filter.

    :param voc_root_folder: root folder of the PASCAL dataset
    :return:
    """

    log("Building list of filtered filenames.", lvl=2)

    # Get annotation files (XML)
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []

    # Go over all annotation files
    for a_f in annotation_files:

        # Parse each file
        tree = ET.parse(os.path.join(annotation_folder, a_f))

        # Check for presence of each of the filter classes
        check = [tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]

        # If any of them are present, add it to the list (and disregard the extension)
        if np.any(check):
            filtered_filenames.append(a_f[:-4])

    return filtered_filenames


def pipeline(image_dim=214,
             filter=CLASSES,
             class_data=True,
             segm_data=False,
             mono=True,
             voc_root_folder="../data/VOCdevkit/") -> (dict, dict):
    """
    Pipeline to extract training and validation datasets from full dataset.

    :param image_dim: images will be resized to image_dim x image_dim pixels
    :param filter: classes to take into account (out of total of 20)
    :param class_data: load classification data
    :param segm_data: load segmentation data
    :param voc_root_folder: root folder of PASCAL dataset
    :return: data dicts
    """

    if not (class_data or segm_data):
        raise Exception("No data requested! Choose classification data, segmentation data, or both.")

    log("Preparing data: image dimension {imdim}x{imdim}.".format(imdim=image_dim), lvl=1)

    # Make folders, should they not exist yet
    make_folders("data", "scripts",
                 os.path.join('data', 'image_dim_' + str(image_dim)))

    # Try to load data sets locally
    data_path = os.path.join(os.pardir, 'data', 'image_dim_' + str(image_dim), "data_" + str(image_dim) + ".npy")
    data_segm_path = os.path.join(os.pardir, 'data', 'image_dim_' + str(image_dim),
                                  "data_segm_" + str(image_dim) + ('mono' if mono else '') + ".npy")

    # Build (x,y) for TRAIN/TRAINVAL/VAL (classification)
    log("Building filtered dataset.", lvl=2)

    classes_folder = os.path.join(voc_root_folder, "VOC2009", "ImageSets", "Main")
    classes_files = os.listdir(classes_folder)

    # Only take training and validation images that pass filter
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if
                   filt in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if
                 filt in c_f and '_val.txt' in c_f]

    # If classification data is requested
    if class_data:

        # Try to load the data from disk
        log("Getting classification data.", lvl=3)
        try:

            data = np.load(file=data_path).item()
            log("Loaded classification data.", lvl=2)

            train_images = data['train_images']
            val_images = data['val_images']

        except (NameError, FileNotFoundError):

            log("Failed to load one of the {}-sized datasets. Rebuilding.".format(image_dim), lvl=1)
            x_train, y_train, train_images = build_classification_dataset(list_of_files=train_files,
                                                                          image_dim=image_dim,
                                                                          filter=filter,
                                                                          voc_root_folder=voc_root_folder)
            log("Extracted {} training images from {} classes.".format(x_train.shape[0],
                                                                       y_train.shape[1]),
                lvl=3)
            x_val, y_val, val_images = build_classification_dataset(list_of_files=val_files, image_dim=image_dim,
                                                                    filter=filter,
                                                                    voc_root_folder=voc_root_folder)
            log("Extracted {} validation images from {} classes.".format(x_val.shape[0], y_train.shape[1]), lvl=3)

            # Keep random sample as test
            test_indices = rnd.sample(range(x_val.shape[0]), k=len(x_val) // 2)

            x_test = x_val[test_indices]
            y_test = y_val[test_indices]

            x_val = x_val[np.invert(test_indices)]
            y_val = y_val[np.invert(test_indices)]

            data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_val': x_val,
                'y_val': y_val,
                'x_test': x_test,
                'y_test': y_test,
                'train_images': train_images,
                'val_images': val_images
            }

            # Save locally, because this takes a while
            log("Storing classification data.", lvl=3)
            np.save(data_path, data)

    else:
        data = {}

    # If segmentation data is requested
    if segm_data:

        # Try to load segmentation data from disk
        log("Getting segmentation data.", lvl=3)
        try:

            data_segm = np.load(file=data_segm_path).item()
            log("Loaded segmentation data.", lvl=2)

            # ... else build datasets
        except (FileNotFoundError, NameError):

            if not class_data:
                raise Exception("Failed to load segmentation data. Set 'class_data' to true to build datasets!")

            log("Failed to load segmentation data. Rebuilding.", lvl=3)
            data_segm = build_segmentation_dataset(train_list=train_images,
                                                   val_list=val_images,
                                                   image_dim=image_dim, voc_root_folder=voc_root_folder,
                                                   mono=mono)
            log("Storing segmentation data.", lvl=3)
            np.save(data_segm_path, data_segm)

    else:
        data_segm = {}

    # Return data
    return data, data_segm


################
# Alter images #
################

def threshold(x: np.ndarray, threshold=.02, mono=True) -> np.ndarray:
    """
    Threshold images (segmentation images to black and white).

    (Interesting: Otsu threshold - http://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/)
    :param x: image array
    :param threshold: determines whether pixel is made black or white
    :return: new image array
    """

    # Convert to grayscale if needed
    try:
        x_new = rgb2gray(x)
    except:
        x_new = x

    # See where pixel values surpass the threshold
    super_threshold_indices = x_new > threshold

    # Set all these pixels to white, and the rest to black
    x_new[super_threshold_indices] = 1.0
    x_new[np.invert(super_threshold_indices)] = 0.0

    # Convert back to color
    if mono:
        return x_new[:, :, :, np.newaxis]
    else:
        return (gray2rgb(x_new))


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
        image_dim = round(sqrt(image_size / 3))
        return x.reshape((len(x), image_dim, image_dim, 3))
    else:
        log("Failed to lift x - already expanded.", lvl=3)
        return x


#############
# Visualize #
#############

def show_some(image_list: list, subtitles=None, n=10, random=True, title="", save=False, save_name="test"):
    """
    Show some images in the training dataset.

    :param images_list: a list of image arrays
    :param images_types: list of titles for the rows
    :param n: number of images to show (per array)
    :param title: add title to plot if requested
    :return: None
    """

    if not subtitles:
        subtitles = ["" for i in range(len(image_list))]

    if len(image_list) != len(subtitles) and len(subtitles) != 0:
        raise Exception("Either don't pass subtitles, or make sure each image row has a title.")

    # Get indices
    indices = rnd.sample(range(image_list[0].shape[0]), n) if random else range(n)

    # Show corresponding images
    rows = len(image_list)
    plt.figure(figsize=(5 + n, 3 * rows))

    for row, images in enumerate(image_list):

        for col, idx in enumerate(indices):

            image = images[idx]

            if image.shape[2] == 1:
                image = gray2rgb(image[:, :, 0])

            plt.subplot(rows, n, (n * row) + col + 1)
            plt.axis('off')
            plt.imshow(image)
            if col == 0:
                plt.title(subtitles[row], ha='left', family='serif', weight='bold', style='italic')

    plt.tight_layout()
    plt.suptitle(title, weight='bold')

    # Save plot if requested
    if save:
        plot_path = os.path.join(os.pardir, "models", "segmentation", "plots", save_name + ".png")
        plt.savefig(plot_path)

    plt.show()


def plot_model_histories(histories: dict, image_dim: int, compression_factor: int, save=True, plot=True) -> plt.Axes:
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
                    y_limit = (.7, .9)

                else:
                    key = 'loss' if col == 0 else 'val_loss'
                    title = '{} loss'.format(train_or_val)
                    y_label = 'Loss (binary cross entropy)'
                    y_limit = (.2, .7)

                # Plot training & validation accuracy values
                ax.plot(history.history[key], label="Model: {}".format(label), color=colors[color_counter])

                # Add vertical line to indicate early stopping
                ax.axvline(x=n_epochs, linestyle='--', color=colors[color_counter])

                # Set a title, the correct y-label, and the y-limit
                ax.set_title(title, fontdict={'fontweight': 'semibold'})
                ax.set_ylabel(y_label)
                ax.set_ylim(y_limit)

                color_counter += 1

            if row == 1: ax.set_xlabel('Epoch')
            ax.set_xlim(0, 200)
            ax.legend(loc='best')

    # Title
    plt.suptitle(
        "Training histories of classifier models (image dim {} - compression {})".format(image_dim, compression_factor),
        fontweight='bold')

    # Build model path
    evaluation_label = 'histories_im_dim{}comp{}'.format(image_dim, compression_factor)
    plot_path = os.path.join(os.pardir, "models", "autoencoders", "plots", evaluation_label + ".png")

    # Show 'n tell
    if save: fig.savefig(plot_path, dpi=fig.dpi)
    if plot: plt.show()

    return ax


########
# MAIN #
########

if __name__ == '__main__':
    # Let's go
    log("DATA PREPARATION", title=True)

    # Set logging parameters
    hlp.LOG_LEVEL = 3

    # Get data
    data, _ = pipeline(image_dim=48,
                       class_data=True,
                       segm_data=True,
                       mono=True)

    x_tr, y_tr = data['x_train'], data['y_train']
    x_va, y_va = data['x_val'], data['y_val']
    x_te, y_te = data['x_test'], data['y_test']

    train_images = data['train_images']

    voc_root_folder = "../data/VOCdevkit/"
    segm_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationObject/")
    segm_files = os.listdir(segm_folder)

    show_some(image_list=[x_te], n=10, random=False, title="Test images")
    show_some(image_list=[x_tr], n=10, random=False, title="Test images")

    _, data_segm = pipeline(image_dim=48, class_data=True, segm_data=True)

    x_val = data_segm['x_val']
    y_val = data_segm['y_val']
    y_val_source = data_segm['y_val_source']

    show_some(image_list=[x_val, y_val_source, y_val], subtitles=["Original", "Segmented with threshold", "Segmented"],
              n=6, random=True, title='Segmentation images')
