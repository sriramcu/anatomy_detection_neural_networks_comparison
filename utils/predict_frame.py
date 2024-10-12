"""
Module to predict class label of a frame with a prediction probability
"""
import os

from utils.constants import DISABLE_TF_LOGGING

if DISABLE_TF_LOGGING:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# To disable tensorflow debugging log messages 

import numpy as np
from keras.utils import load_img, img_to_array
import PIL
import cv2


def test_img(filename, img_width, img_height, model, datagen, class_names):
    """
    Runs model prediction on image file by calling the test_frame() function

    Args:
        filename (str): absolute path to image file
        img_width (int): width of image
        img_height (int): height of image
        model (keras.Model): loaded keras model
        datagen (ImageDataGenerator): Keras ImageDataGenerator object containing transformations to run on the image
        class_names (list): list of classes predicted by the model

    Returns:
        tuple: True, Predicted Class Name, Prediction probability % (returned from test_frame() function)
    """

    try:
        img = load_img(filename, target_size=(img_width, img_height))
    except PIL.UnidentifiedImageError:
        print(f"{filename} couldn't be processed by load_img")
        return False, None

    test_image = img_to_array(img)

    return test_frame(test_image, model, datagen, class_names)


def test_frame(test_image, model, datagen, class_names):
    """
    Runs model prediction on image array

    Args:
        test_image (np.ndarray): input image array in BGR
        model (keras.Model): loaded keras model
        datagen (ImageDataGenerator): Keras ImageDataGenerator object containing transformations to run on the image
        class_names (list): list of classes predicted by the model

    Returns:
        tuple: True, Predicted Class Name, Prediction probability %
    """

    original_test_image = test_image.copy()

    try:

        test_image = np.expand_dims(test_image, axis=0)
        test_image = datagen.standardize(test_image)
        # test_image.shape
        # test_image = test_image/255
        # images = np.vstack([test_image])
        predicted_classes = model.predict(test_image, batch_size=10)


    except ValueError:
        print("Dimension of test image doesn't match input dimension of network")
        h = int(input("Enter input height: "))
        w = int(input("Enter input width: "))

        test_image = cv2.resize(original_test_image, (h, w))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = datagen.standardize(test_image)

        try:
            predicted_classes = model.predict(test_image)

        except ValueError:
            test_image = np.squeeze(test_image)

            test_image = cv2.resize(test_image, (h, w))

            test_image = np.expand_dims(test_image, axis=0)
            predicted_classes = model.predict(test_image)

    predicted_class = class_names[np.argmax(predicted_classes)]
    rounded_prob = round(np.amax(predicted_classes) * 100, 2)

    return True, predicted_class, rounded_prob
