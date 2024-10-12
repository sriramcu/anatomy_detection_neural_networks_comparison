"""
Standalone program to split a dataset into training and validation 
"""
import argparse
import os
import shutil


def main():
    """
    Main function to split a dataset into training and validation- specify original train directory\
        and train-val split ratio (0-1) supplied via the cmd line

    Raises:
        FileNotFoundError: If train directory does not exist
        ValueError: If train directory name is not 'train'
        ValueError: If train-val split ratio is not between 0 and 1
    """

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("-train_dir", type=str, required=True, help="Train directory- REQUIRED")
    ap.add_argument("-val_split", type=float, required=True, help="Train-val split- REQUIRED")

    args = vars(ap.parse_args())

    # Change working directory of the script to the folder containing 'train' and 'val' folders
    train_dir = os.path.abspath(args["train_dir"])

    if not os.path.isdir(train_dir):
        raise FileNotFoundError("Train Directory cannot be found")

    if os.path.basename(train_dir) != "train":
        raise ValueError("Train directory name must be 'train'")

    os.chdir(os.path.dirname(train_dir))

    os.makedirs("val", exist_ok=True)

    train_categories = list(os.listdir(train_dir))

    val_split = float(args["val_split"])

    if val_split <= 0 or val_split >= 1:
        raise ValueError("Train-val split ratio must be between 0 and 1")

    sampling_rate = int(1 / val_split)

    for category in train_categories:
        os.makedirs(os.path.join("val", category), exist_ok=True)
        all_images = list(os.listdir(os.path.join(train_dir, category)))

        for img_num, img_name in enumerate(all_images):
            if img_num % sampling_rate != 0:
                continue

            shutil.move(
                os.path.join(os.getcwd(), 'train', category, img_name),
                os.path.join(os.getcwd(), 'val', category, img_name)
            )


if __name__ == "__main__":
    main()
