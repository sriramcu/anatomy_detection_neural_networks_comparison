"""
Standalone program to evaluate the performance of a model using confusion matrices
"""
import argparse
import csv
import os
import pickle

from keras.models import load_model

from utils.compute_conf_mat_scores import extract_conf_mat, compute_conf_mat_scores
from utils.constants import CHECKPOINTS_DIR, CLASS_LABELS, CONF_CSV_DIR, CONF_PICKLE_DIR, VAL_DIR
from utils.test_images_from_directory import test_images_from_directory


def evaluate_model(root_val_dir, metrics_pickle_filepath):
    """
    Evaluates the performance of a model using confusion matrices

    Args:
        root_val_dir (str): Root directory containing classwise segregated images for testing/validation
        metrics_pickle_filepath (str): Path to pickle file generated after model training
    """

    results = {}

    classes = CLASS_LABELS

    f = open(metrics_pickle_filepath, 'rb')

    _ = pickle.load(f)
    params = pickle.load(f)

    f.close()

    custom_preprocessing = params[-3]

    checkpoints_filename = os.path.basename(params[-1])

    checkpoints_filepath = os.path.join(CHECKPOINTS_DIR, checkpoints_filename)

    if not os.path.isfile(checkpoints_filepath):
        raise FileNotFoundError(f"Checkpoints file '{checkpoints_filepath}' not found!")

    cp_basename = checkpoints_filename.split(".")[0]

    conf_pickle = os.path.join(CONF_PICKLE_DIR, f"{cp_basename}.pkl")

    conf_csv = os.path.join(CONF_CSV_DIR, f"{cp_basename}.csv")

    model = load_model(checkpoints_filepath)

    for actual_class_name in classes:
        test_dir = os.path.join(root_val_dir, actual_class_name)

        results_dict = test_images_from_directory(model,
                                                  test_dir=test_dir,
                                                  custom_preprocessing=custom_preprocessing
                                                  )

        for possible_class_name in classes:
            if possible_class_name not in results_dict.keys():
                results_dict[possible_class_name] = 0

        results[actual_class_name] = results_dict

    f = open(conf_pickle, 'wb')
    pickle.dump(results, f)
    f.close()

    # Columns-predicted class names, Rows- actual class names

    with open(conf_csv, 'w') as f_object:

        writer_object = csv.writer(f_object)

        header_row = ["Class"]
        header_row.extend(classes)

        writer_object.writerow(header_row)

        for actual_class_name in classes:
            row = [actual_class_name]
            actual_class_dict = results[actual_class_name]
            for predicted_class in classes:
                row.append(actual_class_dict[predicted_class])

            writer_object.writerow(row)

    conf_mat = extract_conf_mat(conf_pickle)
    compute_conf_mat_scores(conf_mat, conf_pickle=conf_pickle)


def main():
    """
    Runs the evaluate() function based on command line arguments (only train pickle file path is mandatory)
    """

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-dir",
        "--root_val_dir",
        default=VAL_DIR,
        help="Root directory containing classwise segregated images for testing/validation, default: %(default)s"
    )

    requiredNamed = ap.add_argument_group('required named arguments')

    requiredNamed.add_argument(
        "-pfile", "--train_pickle_file", required=True,
        help="Path to pickle file generated after model training (REQUIRED)"
    )

    args = vars(ap.parse_args())

    root_val_dir = os.path.abspath(args["root_val_dir"])

    metrics_pickle_filepath = os.path.abspath(args["train_pickle_file"])

    evaluate_model(root_val_dir, metrics_pickle_filepath)


if __name__ == "__main__":
    main()
