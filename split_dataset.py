"""
Standalone program to split a dataset into training and validation 
"""
import argparse
import os
import random
import shutil

from utils.constants import TRAIN_DIR


def main():
    """
    Main function to split a dataset into training and validation - specify original train directory\
        and train-val split ratio (0-1) supplied via the cmd line

    Raises:
        FileNotFoundError: If train directory does not exist
        ValueError: If train directory name is not 'train'
        ValueError: If train-val split ratio is not between 0 and 1
    """

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-d", "--train_dir", type=str, default=TRAIN_DIR,
                    help="Train directory absolute path (default: %(default)s)")
    ap.add_argument("-v", "--val_split", type=float, default=0.15, help="Train-val split (default: %(default)s)")
    ap.add_argument("-t", "--test_split", type=float, default=0.15, help="Train-test split (default: %(default)s)")
    args = vars(ap.parse_args())

    val_split_ratio = float(args["val_split"])
    test_split_ratio = float(args["test_split"])
    train_dir = os.path.abspath(args["train_dir"])

    original_dir = os.path.abspath(os.getcwd())
    # Change working directory of the script to the folder containing 'train' folder
    if not os.path.isdir(train_dir):
        raise FileNotFoundError("Train Directory cannot be found")
    if os.path.basename(train_dir) != "train":
        raise ValueError("Train directory name must be 'train'")
    os.chdir(os.path.dirname(train_dir))

    print(f"Number of files in train directory before split: {sum([len(files) for r, d, files in os.walk(train_dir)])}")

    if "val" in os.listdir() and "test" in os.listdir():
        print("Dataset has already been split into test and val. Exiting...")
        os.chdir(original_dir)
        return

    if "val" in os.listdir() or "test" in os.listdir():
        os.chdir(original_dir)
        raise ValueError(
            "One of val or test exists but not both. Please move all data back into train folder and re-run the program")

    if val_split_ratio <= 0 or val_split_ratio >= 1 or test_split_ratio <= 0 or test_split_ratio >= 1:
        raise ValueError("Train-val and train-test split ratios must be between 0 and 1")

    for category in list(os.listdir(train_dir)):
        os.makedirs(os.path.join("val", category))
        os.makedirs(os.path.join("test", category))
        all_images_in_a_category = list(os.listdir(os.path.join(train_dir, category)))
        random.shuffle(all_images_in_a_category)
        num_val_images = int(val_split_ratio * len(all_images_in_a_category))
        num_test_images = int(test_split_ratio * len(all_images_in_a_category))
        for img_num, img_name in enumerate(all_images_in_a_category):
            if img_num < num_val_images:
                shutil.move(
                    os.path.join(os.getcwd(), 'train', category, img_name),
                    os.path.join(os.getcwd(), 'val', category, img_name)
                )
            elif img_num < num_val_images + num_test_images:
                shutil.move(
                    os.path.join(os.getcwd(), 'train', category, img_name),
                    os.path.join(os.getcwd(), 'test', category, img_name)
                )

    print("Splitting complete.")
    print(f"Number of files in train directory: {sum([len(files) for r, d, files in os.walk(train_dir)])}")
    print(f"Number of files in val directory: {sum([len(files) for r, d, files in os.walk('val')])}")
    print(f"Number of files in test directory: {sum([len(files) for r, d, files in os.walk('test')])}")
    os.chdir(original_dir)


if __name__ == "__main__":
    main()
