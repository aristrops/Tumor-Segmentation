import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

