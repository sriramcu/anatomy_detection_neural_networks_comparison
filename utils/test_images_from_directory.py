import pickle
from keras.models import load_model
from my_datagen import return_my_datagen
import sys
import os
import time
from pathlib import Path
import argparse

from custom_preprocess import custom_preprocess
from predict_frame import test_img
from constants import *


def test_images_from_directory(model, test_dir, class_names = [], custom_preprocessing=True):
    """
    Test images present either directly or recursively (class-wise) inside a directory

    Args:
        model (keras.Model): Loaded keras model
        test_dir (str): root directory where test images are located
        class_names (list, optional): model class names; if empty, classes taken from constants.py . Defaults to [].
        custom_preprocessing (bool, optional): specifies if custom preprocessing used for prediction. Defaults to True.

    Returns:
        dict: Dictionary whose keys are class names and values are no. of predictions made per class
    """
    

    test_dir_path = Path(test_dir)
    if class_names == []:
        # This will no longer be the default case since it is now taken from constants.py
        class_names = sorted(os.listdir(test_dir_path.resolve().parent))

    val_datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="prediction") 
        
    
    img_width, img_height = TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT

    print(f"Testing images located in {test_dir}")
    counter = 0
    results_dict = {}
    start_time = time.time()
    
    test_dir_size = len(os.listdir(test_dir))
    for filename_img in os.listdir(test_dir):
        
        filename = os.path.abspath(os.path.join(test_dir,filename_img))       
        
        successful, predicted_class, _ = test_img(filename, img_width, img_height, 
                                               model, val_datagen, class_names)
        
        if not successful:
            print(f"Could not process image {filename}")
            continue

        if predicted_class not in results_dict.keys():
            results_dict[predicted_class] = 1
        else:
            results_dict[predicted_class] += 1

        counter += 1    
        
        if counter % 100 == 0:
            print(f"{counter} out of {test_dir_size} files processed!")


    time_taken = time.time() - start_time
    time_taken = round(time_taken,2)
    
    print(f"{counter} images processed in {time_taken} seconds, at a rate of "
          f"{round(counter/time_taken,2)} images per second.")
    
    for predicted_class in results_dict.keys():
        num_predictions = results_dict[predicted_class]
        percentage_predicted = round(100*num_predictions/counter,2)
        print(f"{predicted_class} = {num_predictions} predictions ({percentage_predicted}%)")

    return results_dict


def main():
    """
    Runs test_images_from_directory() function based on test directory & \
        training pickle file supplied from cmd line. Class names taken from constants.py
    """
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            
    requiredNamed = ap.add_argument_group('required named arguments')

    requiredNamed.add_argument("-test_directory", required=True, help=" ", type=str)
    
    requiredNamed.add_argument("-train_pickle_file", required=True, help=" ")    
    
    args = vars(ap.parse_args())    

    class_names = CLASS_LABELS

    print(f"Classes being predicted by this model are: {class_names}")

    
    metrics_pickle_filepath = args["train_pickle_file"]

    f = open(metrics_pickle_filepath, 'rb')
    
    _ = pickle.load(f)
    params = pickle.load(f)
    
    f.close()
    
    checkpoints_filename = os.path.basename(params[-1])

    checkpoints_filepath = os.path.join(CHECKPOINTS_DIR, checkpoints_filename)

    if not os.path.isfile(checkpoints_filepath):
        raise FileNotFoundError(f"Checkpoints file '{checkpoints_filepath}' not found!")

    model = load_model(checkpoints_filepath)
    
    test_images_from_directory(model, test_dir=args["test_directory"], 
                               class_names=class_names,
                               custom_preprocessing=params[-2]
                               )


if __name__ == "__main__":
    main()