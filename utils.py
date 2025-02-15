import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

#load the breast cancer image/mask pairs
def create_list():
    directory = "Datasets/Dataset_BUSI_with_GT"
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


#load the brain cancer image/mask pairs
def create_list_brain():
    directory = "Datasets/Brain_tumor_dataset"
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


#functions to map to the tensorflow dataset
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

#function to augment the training set
def augment_image_and_mask(image, mask):

    #horizontal flip
    flip_hor = tf.random.uniform([]) > 0.5
    image = tf.cond(flip_hor, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip_hor, lambda: tf.image.flip_left_right(mask), lambda: mask)

    #vertical flip
    flip_ver = tf.random.uniform([]) > 0.5
    image = tf.cond(flip_ver, lambda: tf.image.flip_up_down(image), lambda: image)
    mask = tf.cond(flip_ver, lambda: tf.image.flip_up_down(mask), lambda: mask)

    #rotation of -20° to +20°
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    #brightness change
    image = tf.image.random_brightness(image, max_delta=0.1)

    #contrast change
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    #crop and resize
    original_shape = tf.shape(image)
    crop_size = tf.cast(original_shape[:2], tf.float32) * 0.9  # Convert shape to float before multiplication
    crop_size = tf.cast(crop_size, tf.int32)  # Convert back to int
    image = tf.image.resize_with_crop_or_pad(image, crop_size[0], crop_size[1])
    mask = tf.image.resize_with_crop_or_pad(mask, crop_size[0], crop_size[1])
    image = tf.image.resize(image, (original_shape[0], original_shape[1]))
    mask = tf.image.resize(mask, (original_shape[0], original_shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, mask


#function to create the tensorflow dataset
def create_dataset(pairs, augment=False):
    image_path = [pair[0] for pair in pairs]
    mask_path = [pair[1] for pair in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((image_path, mask_path))

    dataset = dataset.map(load_images_and_masks)

    if augment:
        dataset = dataset.map(augment_image_and_mask)

    return dataset


#function to split the dataset into train/test/val
def split_dataset(image_and_mask_list):
    #split into training and validation/test
    train_pairs, val_test_pairs = train_test_split(image_and_mask_list, test_size=0.4, random_state=42)

    #split validation and test set
    val_pairs, test_pairs = train_test_split(val_test_pairs, test_size=1/2, random_state=42)

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

def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coefficient(y_true, y_pred, smooth)

def bce_dice_loss(y_true, y_pred, smooth=1e-6):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred, smooth)

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return 1 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)