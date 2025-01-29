import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#load the breast cancer image/mask pairs
def create_list():
    directory = "Dataset_BUSI_with_GT"
    categories = ["benign", "malignant"]

    image_and_mask_list = []

    for category in categories:
        path = os.path.join(directory, category)
        #get all files in the folder
        files = os.listdir(path)

        #separate images and masks
        images = [f for f in files if not f.endswith("_mask*.png")]
        masks = [f for f in files if f.endswith("_mask.png")]

        #match images with their masks
        for image in images:
            mask_name = f"{os.path.splitext(image)[0]}_mask.png"
            if mask_name in masks:
                image_path = os.path.join(path, image)
                mask_path = os.path.join(path, mask_name)
                image_and_mask_list.append((image_path, mask_path))
    return image_and_mask_list

#load the skin cancer image/mask pairs
def create_list_skin(image_directory, mask_directory): #different for training, validation and test
    image_and_mask_list = []
    images = os.listdir(image_directory)
    masks = os.listdir(mask_directory)
    for image in images:
        if not image.endswith("_superpixels.png"):
            mask_name = f"{os.path.splitext(image)[0]}_segmentation.png"
            if mask_name in masks:
                image_path = os.path.join(image_directory, image)
                mask_path = os.path.join(mask_directory, mask_name)
                image_and_mask_list.append((image_path, mask_path))
    return image_and_mask_list


#load the rectal cancer image/mask pairs
def create_list_brain():
    directory = "Brain_tumor_dataset"

    image_and_mask_list = []

    image_path = os.path.join(directory, "images")
    mask_path = os.path.join(directory, "masks")

    #get all files in both folders
    image_files = os.listdir(image_path)
    mask_files = os.listdir(mask_path)

    #match images with their masks
    common_names = set(image_files) & set(mask_files)

    for file in common_names:
        image_and_mask_list.append((os.path.join(image_path, file), os.path.join(mask_path, file)))

    return image_and_mask_list

def load_images_and_masks(image_path, mask_path):
    image_size = (256, 256)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, image_size)

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

def create_dataset(pairs):
    image_path = [pair[0] for pair in pairs]
    mask_path = [pair[1] for pair in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((image_path, mask_path))

    dataset = dataset.map(lambda image, mask: load_images_and_masks(image, mask), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def split_dataset(image_and_mask_list):
    #split into training and validation/test
    train_pairs, val_test_pairs = train_test_split(image_and_mask_list, test_size=0.4, random_state=42)

    #split validation and test set
    val_pairs, test_pairs = train_test_split(val_test_pairs, test_size=0.5, random_state=42)

    return train_pairs, val_pairs, test_pairs


#functions to compute useful metrics
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    #flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    #compute the intersection
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)

    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    #flatten the tensors
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    #compute the intersection and the union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection

    return (intersection + smooth) / (union + smooth)

