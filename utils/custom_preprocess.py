"""
Module that defines custom preprocessing used for Anatomy Detection
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import skimage

from utils.constants import TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT


def edge_removal(image, threshold=50, max_black_ratio=0.7, mode="hsv"):
    """
    Function that removes horizontal and vertical black lines (called 'edges') from an image

    Args:
        image (np.ndarray): input image in the color space as defined by the "mode" parameter
        threshold (int, optional): int value (0-255), if all 3 pixel channels are < threshold, pixel is black. Defaults to 50.
        max_black_ratio (float, optional): Max ratio of permissible "black" pixels in a row or column. Defaults to 0.7.
        mode (str, optional): color space of the input image. Defaults to "hsv".
    
    Returns:
        np.ndarray: output image in the same color space as the input image
    """

    def pixels_should_be_conserved(pixels) -> bool:
        """
        Checks a row or column of pixels whether it should be removed based on threshold pixel value and max black ratio

        Args:
            pixels (np.ndarray): 1D array of pixels (corresponding to a row or column of the image)

        Returns:
            bool: whether row/column should be preserved
        """

        if mode == "hsv":
            black_pixel_count = (pixels[:, 2] <= threshold).sum()

        elif mode == "yuv":
            black_pixel_count = (pixels[:, 0] <= threshold).sum()

        elif mode == "grayscale":
            black_pixel_count = (pixels <= threshold).sum()
        else:
            # mode is bgr
            black_pixel_count = (pixels <= threshold).all(axis=1).sum()

        pixel_count = len(pixels)

        return pixel_count > 0 and black_pixel_count / pixel_count <= max_black_ratio

    num_rows, num_columns, _ = image.shape
    preserved_rows = [r for r in range(num_rows) if pixels_should_be_conserved(image[r, :, :])]
    preserved_columns = [c for c in range(num_columns) if pixels_should_be_conserved(image[:, c, :])]

    image = image[preserved_rows, :, :]
    image = image[:, preserved_columns, :]

    return image


def clahe(input_color_img, clip_limit: float, tile_grid_size: tuple):
    """
    Applies CLAHE histogram equalisation technique to an image

    Args:
        input_color_img (np.ndarray): Input image
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization

    Returns:
        np.ndarray: output image
    """
    clahe_model = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    color_image_b = clahe_model.apply(input_color_img[:, :, 0])
    color_image_g = clahe_model.apply(input_color_img[:, :, 1])
    color_image_r = clahe_model.apply(input_color_img[:, :, 2])
    input_color_img = np.stack((color_image_b, color_image_g, color_image_r), axis=2)
    return input_color_img


def magva(image: np.ndarray):
    """
    Applies the novel MAGVA (Cogan et al.) contrast enhancement technique to an image

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: output image
    """

    img_mean = np.mean(image)
    desired_mean = 90
    margin = 1

    lower_bound = desired_mean - margin
    upper_bound = desired_mean + margin

    while img_mean < lower_bound or img_mean > upper_bound:
        gamma = math.log(desired_mean / 255) / math.log(img_mean / 255)
        image = skimage.exposure.adjust_gamma(image, gamma)
        img_mean = np.mean(image)

    return image


def apply_low_pass_filter(image):
    """
    Smoothens an image by passing it through a low pass band filter

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: output image
    """
    kernel = np.array([[0.1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 0.1]]) / 1.8
    image = cv2.filter2D(image, cv2.CV_64F, kernel)
    return image


def custom_preprocess(image, clip_limit: float = 1.0, tile_grid_size: tuple = (2, 2), image_format="rgb"):
    """
    Custom preprocessing function applied on an image array using a sequence of techniques defined by aforementioned functions

    Args:
        image (np.ndarray): input image in the color space as defined by "image_format" argument
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 1.0.
        tile_grid_size (tuple, optional): Size of grid for histogram equalization. Defaults to (2,2).
        image_format (str, optional): Color space of input image, either "bgr" or "rgb". Defaults to "rgb".

    Returns:
        np.ndarray: output image
    """

    img_was_squeezed = False
    # flag variable to check whether the dimensions of the input image were squeezed 
    # (happens in image evaluation but not training)

    # we check the image color space and then convert the image to the appropriate color space and datatype
    if image_format == "rgb":
        # usually happens in the ImageDataGenerator function since keras modules load image in RGB and as float
        if image.ndim == 4:
            image = np.squeeze(image, axis=0)
            img_was_squeezed = True
        image = np.asarray(image)
        image = np.uint8(image)  # to make img array compatible with opencv functions likle cvtcolor
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Now image array will have 3 dimensions in the YUV color space, where array datatype is uint8

    image = edge_removal(image, threshold=37, mode="yuv")
    y_component = image[:, :, 0]
    u_component = image[:, :, 1]
    v_component = image[:, :, 2]

    # Now, for YUV color space, we only need to process the 'Y' component with MAGVA and the low pass filter
    y_component = magva(y_component)
    # we could also use CLAHE instead; MAGVA proved to have better results

    y_component = apply_low_pass_filter(y_component)
    y_component = y_component.astype(u_component.dtype)
    # Since cv2.merge needs all channels to have the same dtype & MAGVA/Low pass could change the dtype during processing
    yuv_image = cv2.merge((y_component, u_component, v_component))
    image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    image = cv2.resize(image, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)

    if img_was_squeezed:
        # if image originally had 4 dims before we squeezed it prior to the 1st YUV conversion, reinstate the fourth dimension
        image = np.expand_dims(image, axis=0)

    image = np.float32(image)  # since keras image prediction/training requires float datatype
    return image


def custom_preprocess_image_file(img_path, clip_limit: float, tile_grid_size: tuple):
    """
    Run custom preprocessing on image file by loading it and passing it to the function. \
        Generates output image after custom preprocessing so that programmer can tweak parameters/\
            techniques used for preprocessing by observing and comparing results for an individual input img 

    Args:
        img_path (str): absolute path to input image file
        clip_limit (float): Threshold for contrast limiting
        tile_grid_size (tuple): Size of grid for histogram equalization

    Raises:
        FileNotFoundError: if input image file is not found
    """

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Input image file {img_path} not found")

    img_arr = cv2.imread(img_path)  # BGR
    img_arr = cv2.resize(img_arr, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT))
    img_arr = custom_preprocess(img_arr, clip_limit, tile_grid_size, image_format="bgr")

    img_path_without_ext = "".join(img_path.split(".")[:-1])
    img_path_without_ext += f"_{clip_limit}_{tile_grid_size}"
    new_img_path = img_path_without_ext + "." + img_path.split(".")[-1]
    cv2.imwrite(new_img_path, img_arr)


def main():
    """
    Main function to handle cmd line arguments to run the custom_preprocess_image_file() function.\
        grid_size is an int parameter from the cmd line that's converted into a tuple (grid_size, grid_size) passed to the function
    """

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-image_filepath", type=str, required=True, help="REQUIRED")
    ap.add_argument("-clip_limit", type=float, default=1.0, help=" ")
    ap.add_argument("-grid_size", type=int, default=2, help=" ")
    args = vars(ap.parse_args())

    grid_size = (int(args["grid_size"]), int(args["grid_size"]))
    custom_preprocess_image_file(args["image_filepath"], args["clip_limit"], grid_size)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Time taken = {time.time() - start_time}")
