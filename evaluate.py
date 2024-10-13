"""
Standalone program to evaluate the performance of a model using confusion matrices
"""
import argparse
import csv
import os
import pickle
import time
from pathlib import Path

import efficientnet.keras as efn
import numpy as np
import pandas as pd
from keras.models import load_model

from utils.constants import CHECKPOINTS_DIR, CLASS_LABELS, CONF_CSV_DIR, TRAIN_IMAGE_WIDTH, \
    TRAIN_IMAGE_HEIGHT, TEST_DIR
from utils.generate_datagen import get_datagen_obj
from utils.predict_frame import test_image_file


def test_images_from_directory(model, test_dir, custom_preprocessing=True):
    """
    Test images present either directly or recursively (class-wise) inside a directory, supports single and multiclass

    Args:
        model (keras.Model): Loaded keras model
        test_dir (str): root directory where test images are located
        custom_preprocessing (bool, optional): specifies if custom preprocessing used for prediction. Defaults to True.

    Returns:
        results_dict (dict): Dictionary whose keys are class names and values are no. of predictions made per class
    """
    test_class_names = sorted(os.listdir(Path(test_dir).resolve().parent))  # will be a subset of CLASS_LABELS
    val_datagen = get_datagen_obj(custom_preprocessing=custom_preprocessing, mode="prediction")
    img_width, img_height = TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT
    print(f"Testing images located in {test_dir}")
    counter = 0
    results_dict = {}
    start_time = time.time()
    test_dir_size = len(os.listdir(test_dir))

    for filename_img in os.listdir(test_dir):
        test_img_fpath = os.path.abspath(os.path.join(test_dir, filename_img))
        successful, predicted_class, _ = test_image_file(test_img_fpath, img_width, img_height,
                                                         model, val_datagen, test_class_names)
        # probability not needed for creating confusion matrix
        if not successful:
            print(f"Could not process image {test_img_fpath}")
            continue
        if predicted_class not in results_dict.keys():
            results_dict[predicted_class] = 1
        else:
            results_dict[predicted_class] += 1
        counter += 1
        if counter % 100 == 0:
            print(f"{counter} out of {test_dir_size} files processed!")

    time_taken = time.time() - start_time
    time_taken = round(time_taken, 2)

    print(f"{counter} images processed in {time_taken} seconds, at a rate of "
          f"{round(counter / time_taken, 2)} images per second.")

    for predicted_class in results_dict.keys():
        num_predictions = results_dict[predicted_class]
        percentage_predicted = round(100 * num_predictions / counter, 2)
        print(f"{predicted_class} = {num_predictions} predictions ({percentage_predicted}%)")

    return results_dict


def extract_conf_mat(results_dict: dict):
    """
    Extract confusion matrix as a 2D array from a dictionary
    Args:
        results_dict (dict): dictionary whose keys are actual classes and values are model prediction dictionaries
    Returns:
        list: confusion matrix
    """
    conf_mat = []
    sorted_actual_class_names = sorted(results_dict.keys(), key=lambda x: x.lower())
    for actual_class in sorted_actual_class_names:
        row_dict = results_dict[actual_class]
        sorted_model_prediction_classes = sorted(row_dict.keys(), key=lambda x: x.lower())
        row_values = []
        for model_predicted_class in sorted_model_prediction_classes:
            row_values.append(row_dict[model_predicted_class])
        conf_mat.append(row_values)
    return conf_mat


def compute_conf_mat_scores(conf_mat):
    """
    Calculates confusion matrix scores like accuracy, precision, etc. from the confusion matrix 2d array

    Args:
        conf_mat (list): 2d list of confusion matrix

    Returns:
        percentage_metrics (dict): dictionary of confusion matrix scores
    """

    df_cm = pd.DataFrame(conf_mat, range(6), range(6))

    print(df_cm)

    TP = np.diag(conf_mat)
    FP = np.sum(conf_mat, axis=0) - TP
    FN = np.sum(conf_mat, axis=1) - TP

    num_classes = 6
    TN = []
    for i in range(num_classes):
        tmp_2d_arr = np.delete(conf_mat, i, 0)  # delete ith row
        tmp_2d_arr = np.delete(tmp_2d_arr, i, 1)  # delete ith column
        TN.append(
            sum(sum(tmp_2d_arr)))  # sum of flattened array which contains all values except for ith diagonal element

    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)  # recall
    specificity = TN / (TN + FP)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    percentage_metrics = {}

    def average_rounded_percentage(arr):
        return round(np.mean(arr) * 100, 2)

    percentage_metrics['Precision'] = average_rounded_percentage(precision)
    percentage_metrics['Accuracy'] = average_rounded_percentage(accuracy)
    percentage_metrics['Recall (Sensitivity)'] = average_rounded_percentage(sensitivity)
    percentage_metrics['Specificity'] = average_rounded_percentage(specificity)
    percentage_metrics['F1 score'] = average_rounded_percentage(f1)
    return percentage_metrics


def evaluate_model(test_dir, metrics_pickle_filepath):
    """
    Evaluates the performance of a model using confusion matrices

    Args:
        test_dir (str): Root directory containing class wise segregated images for testing
        metrics_pickle_filepath (str): Path to pickle file generated after model training
    """
    f = open(metrics_pickle_filepath, 'rb')
    _ = pickle.load(f)
    params = pickle.load(f)
    f.close()

    custom_preprocessing = params[-3]
    checkpoints_filename = os.path.basename(params[-1])
    checkpoints_filepath = os.path.join(CHECKPOINTS_DIR, checkpoints_filename)

    if not os.path.isfile(checkpoints_filepath):
        raise FileNotFoundError(f"Checkpoints file '{checkpoints_filepath}' not found!")

    checkpoints_basename = checkpoints_filename.split(".")[0]
    conf_matrix_csv_fpath = os.path.join(CONF_CSV_DIR, f"{checkpoints_basename}.csv")

    model = load_model(checkpoints_filepath)
    results = {}  # key is actual class label, value is dict of predicted class labels
    for actual_class_label in CLASS_LABELS:
        test_class_dir = os.path.join(test_dir, actual_class_label)
        model_results = test_images_from_directory(model,
                                                   test_dir=test_class_dir,
                                                   custom_preprocessing=custom_preprocessing
                                                   )

        for possible_class_name in CLASS_LABELS:
            if possible_class_name not in model_results.keys():
                model_results[possible_class_name] = 0

        results[actual_class_label] = model_results

    # Columns - predicted class names, Rows - actual class names
    conf_mat = extract_conf_mat(results)  # 2d list
    percentage_metrics = compute_conf_mat_scores(conf_mat)

    with open(conf_matrix_csv_fpath, 'w') as f_object:
        writer_object = csv.writer(f_object)
        header_row = ["Class"]
        header_row.extend(CLASS_LABELS)
        writer_object.writerow(header_row)

        for actual_class_label in CLASS_LABELS:
            row = [actual_class_label]
            actual_class_dict = results[actual_class_label]
            for predicted_class in CLASS_LABELS:
                row.append(actual_class_dict[predicted_class])
            writer_object.writerow(row)

        for key in percentage_metrics.keys():
            writer_object.writerow([key + " = " + str(percentage_metrics[key])])


def main():
    """
    Runs the evaluate() function based on command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--test_dir",
        default=TEST_DIR,
        help="Root directory containing classwise segregated images for testing, default: %(default)s"
    )
    ap.add_argument(
        "-p", "--train_pickle_file", required=True,
        help="Path to pickle file generated after model training (REQUIRED)"
    )
    args = vars(ap.parse_args())
    print(efn.__file__)  # so that reformat code does not remove efn import
    test_dir = os.path.abspath(args["test_dir"])
    metrics_pickle_filepath = os.path.abspath(args["train_pickle_file"])
    evaluate_model(test_dir, metrics_pickle_filepath)


if __name__ == "__main__":
    main()
