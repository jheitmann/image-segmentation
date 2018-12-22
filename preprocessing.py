import cv2
import numpy as np
import os
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import common

def extract_data(image_path, num_images, img_height, as_rgb, *, verbose=True):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].
    Args:
        image_path: image folder path
        num_images: size 0 of returned tensor, ammount of extracted images
        img_height: resized image target width/height
        as_rgb: flag set True when images need to be loaded as rgb
        verbose: named flag set False to disable information printed by this method
    Returns:
        4D tensor [image index, y, x, channels]
    """
    # Loads all the image file paths in img_filenames 
    imgs = []
    img_filenames = [os.path.join(image_path, fn) for fn in os.listdir(image_path)]
    img_filenames.sort()

    for filename in img_filenames[:num_images]:
        if verbose:
            print(f"Loading {filename}")
        # Reads the image into an np.array, resamples it
        img = cv2.imread(filename, as_rgb)
        img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_AREA)
        if not as_rgb:
            img = img[..., np.newaxis]
        # Converts into correct form [0;1] to input in the neuralnet
        img = img.astype('float32')
        img /= common.PIXEL_DEPTH
        imgs.append(img)            

    return np.array(imgs)


def extract_labels(label_path, num_images, img_height, *, verbose=True):
    """
    Extract the labels into a 1-hot matrix [image index, label index].
    Args:
        label_path: path to label folder
        num_images: size 0 of returned tensor, ammount of extracted images
        img_height: resized image target width/height
        verbose: named flag set False to disable information printed by this method
    Returns:
        np.array of the #num_images labels np.arrays
    """
    # Loads all the label file paths in img_filenames 
    gt_imgs = []
    gt_filenames = [os.path.join(label_path, fn) for fn in os.listdir(label_path)]
    gt_filenames.sort()

    for filename in gt_filenames[:num_images]:
        if verbose:
            print (f"Loading {filename}")

        # Reads the image into an np.array, resamples it
        labels = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        labels = cv2.resize(labels, (img_height, img_height))
        labels = labels[..., np.newaxis]

        # Converts into correct form one of {0;1} to input in the neuralnet
        labels = labels.astype('float32')
        labels /= common.PIXEL_DEPTH
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        gt_imgs.append(labels)

    return np.array(gt_imgs)


def get_checkpoint(img_height, rgb, monitor):
    """
    Given the input parameters of utrain.py, create a Keras Checkpoint
    Args:
        img_height: resized image target width/height
        rgb: bool set True when using 3 channels
        monitor: [acc|loss|val_acc|val_loss] name of metric used for keeping checkpoints
    Returns:
        Tuple of filename and the checkpoint itself
    """
    # Creates the complete path ckpt_file of to-be-saved model checkpoint
    file_id = "unet_{}_{}_{}".format("rgb" if rgb else "bw", img_height, str(datetime.now()).replace(':', '_').replace(' ', '_'))
    ckpt_file = os.path.join(common.RESULTS_PATH, file_id + ".hdf5") 
    print("Checkpoint filename:", ckpt_file)

    return file_id, ModelCheckpoint(ckpt_file, monitor=monitor, verbose=1, save_best_only=True)


def convert_01(image, label):
    """
    Converts an img and mask from values in [0;255] to [0;1]
    Args:
        image: The image numpy array
        label: The mask numpy array
    Returns:
        Converted image and label
    """
    image /= float(common.PIXEL_DEPTH)
    label /= float(common.PIXEL_DEPTH)
    # Thresholds the labels to choose one in {0;1}
    label[label <= .5], label[label > .5] = 0, 1
    return image, label


def get_generators(batch_size, train_path, image_folder, mask_folder, data_gen_args, 
    target_size=(400,400), color_mode="rgb", interpolation="lanczos", image_save_prefix="image", 
    mask_save_prefix="mask", save_to_dir=None, shuffle=False, seed=common.SEED):
    """
    Args:
        batch_size: batch_size of generator
        train_path: path to directory containing subdirectories of images and masks
        image_folder: name of subdirectory in train_path containing images
        mask_folder: name of subdirectory in train_path containing masks
        data_gen_args: args dict fed to the ImageDataGenerator objects
        target_size: resizing size for both images and labels
        color_mode: [grayscale|rbg|rgba] the generator will load resp. 1, 3 or 4 channels
        interpolation: [nearest|bilinear|bicubic|lanczos|box|hamming] method for resampling to target_size
        image_save_prefix: save_prefix of flow_from_directory for images
        mask_save_prefix: save_prefix of flow_from_directory for images
        save_to_dir: [None|str] path of directory in which will be saved the generated pictures. None disables saving
        shuffle: bool set to True to shuffle the flow from the the folders
        seed: rng seed used for shuffling and random transformations
    Raises:
        AssertionError: when any subfolder name ends with a separator char (not supported as classes)
    Returns:
        A generator function generating a formated tuple (image, label) of np.array
    """
    
    image_datagen, mask_datagen = ImageDataGenerator(**data_gen_args), ImageDataGenerator(**data_gen_args)

    # Makes flows
    assert not image_folder.endswith(os.path.sep) and not image_folder.endswith('/'),\
        f"The image path {image_folder} must NOT end with separator for some reason (ex: image/ -> image)"
    assert not mask_folder.endswith(os.path.sep) and not mask_folder.endswith('/'),\
        f"The label path {mask_folder} must NOT end with separator for some reason (ex: label/ -> label)"

    # If save_to_dir is provided, will pass save_to_dir+subf to generators, otherwise doesn't pass this param.
    param_save_to = lambda subf: dict(save_to_dir=os.path.join(save_to_dir, subf)) if save_to_dir else {}

    train_image_generator = image_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[image_folder],
        class_mode=None,
        target_size=target_size,
        color_mode=color_mode,
        interpolation=interpolation,
        **param_save_to("train"),
        save_prefix=image_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="training")
    train_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[mask_folder],
        class_mode=None,
        target_size=target_size,
        color_mode="grayscale",
        interpolation=interpolation,
        **param_save_to("train"),
        save_prefix=mask_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="training")
    validation_image_generator = image_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[image_folder],
        class_mode=None,
        target_size=target_size,
        color_mode=color_mode,
        interpolation=interpolation,
        **param_save_to("val"),
        save_prefix="val_"+image_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="validation")
    validation_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[mask_folder],
        class_mode=None,
        target_size=target_size,
        color_mode="grayscale",
        interpolation=interpolation,
        **param_save_to("val"),
        save_prefix="val_"+mask_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="validation")

    # Makes the generator function of tuples using the two flows
    def generator(images, labels):
        for (image, label) in zip(images, labels):
            yield convert_01(image, label)

    return generator(train_image_generator, train_mask_generator), generator(validation_image_generator, validation_mask_generator)
