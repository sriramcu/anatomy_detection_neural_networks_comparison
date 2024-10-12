"""
Module to fetch ImageDataGenerator object used for augmentations in several locations
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from custom_preprocess import custom_preprocess as custom_preprocessing_function
from utils.constants import SAMPLEWISE_NORM


def get_datagen_obj(custom_preprocessing=True, mode="training"):
    """
    Creates and returns ImageDataGenerator object based on training/prediction mode

    Args:
        custom_preprocessing (bool): Determines if custom preprocessing will be used along with standard augmentations
        mode (str, optional): "prediction" only uses basic augmentations. Defaults to "training".

    Returns:
        ImageDataGenerator: object that describes augmentations to be used on train/val dataset
    """
    if custom_preprocessing:
        preprocessing_function = custom_preprocessing_function
    else:
        preprocessing_function = None

    if mode == "training":
        datagen = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            samplewise_std_normalization=SAMPLEWISE_NORM,
            preprocessing_function=preprocessing_function
        )


    else:
        datagen = ImageDataGenerator(
            rescale=1 / 255,
            samplewise_std_normalization=SAMPLEWISE_NORM,
            preprocessing_function=preprocessing_function
        )

    return datagen
