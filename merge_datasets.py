"""
Standalone program to merge train, test and val datasets back into train dataset
"""
import os
import shutil

from utils.constants import TRAIN_DIR, VAL_DIR, TEST_DIR


def main():
    """
    Main function to merge train, test and val datasets back into train dataset

    Raises:
        FileNotFoundError: If any of the train, val or test directories do not exist
    """

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError("Train Directory cannot be found")
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError("Val Directory cannot be found")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError("Test Directory cannot be found")

    for category in list(os.listdir(TRAIN_DIR)):
        for file in os.listdir(os.path.join(VAL_DIR, category)):
            shutil.move(os.path.join(VAL_DIR, category, file), os.path.join(TRAIN_DIR, category))
        for file in os.listdir(os.path.join(TEST_DIR, category)):
            shutil.move(os.path.join(TEST_DIR, category, file), os.path.join(TRAIN_DIR, category))

    shutil.rmtree(VAL_DIR)
    shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    main()