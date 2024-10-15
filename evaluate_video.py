"""
Standalone program that sequentially runs expert annotation and model prediction on a video file
"""
import argparse
import datetime
import os
import pickle
from pathlib import Path
from time import time

import cv2
import efficientnet.keras as efn
from keras.saving.save import load_model

from utils.custom_exponential_decay_class import CustomExponentialDecay
from utils.constants import CHECKPOINTS_DIR, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, CLASS_LABELS
from utils.generate_datagen import get_datagen_obj
from utils.predict_frame import test_frame
from utils.video_annotator import cli_annotation_input, annotate_video, annotate_frame


def evaluate_video(input_video_path, output_video_path, metrics_pickle_filepath):
    """
    Evaluates an input video by calling the predict_frame.test_frame() function frame wise and annotating it
        using the video_annotator module

    Args:
        input_video_path (str): path to input video
        output_video_path (str): path to output annotated video
        metrics_pickle_filepath (str): path to pickle file generated after model training
    """
    start_time = time()
    f = open(metrics_pickle_filepath, 'rb')
    _ = pickle.load(f)
    params = pickle.load(f)
    f.close()

    custom_preprocessing = params[-3]
    checkpoints_filename = os.path.basename(params[-1])
    checkpoints_filepath = os.path.join(CHECKPOINTS_DIR, checkpoints_filename)

    if not os.path.isfile(checkpoints_filepath):
        raise FileNotFoundError(f"Checkpoints file '{checkpoints_filepath}' not found!")

    try:
        model = load_model(checkpoints_filepath)
    except ValueError:
        model = load_model(checkpoints_filepath, custom_objects={'CustomExponentialDecay': CustomExponentialDecay})

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success, frame = vidcap.read()

    Path(output_video_path).unlink(missing_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    datagen = get_datagen_obj(custom_preprocessing=custom_preprocessing, mode="prediction")
    count = 0
    while success:
        success, frame = vidcap.read()
        if not success:
            break
        original_frame = frame.copy()
        count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT))
        if frame.dtype == "uint8":
            frame = frame.astype("float64")
        if count == 1:
            print("Loading CUDNN for model prediction...")
        # run model prediction on current frame and extract the prediction result for annotation purposes
        _, predicted_class, rounded_prob = test_frame(frame, model, datagen, CLASS_LABELS)
        if count == 1:
            print("CUDNN loaded for model prediction, first frame predicted!")
        elif count % 250 == 0:
            print(f"{count} frames of the test video have been predicted so far...")
        prediction_text = f"Model - {predicted_class}:{rounded_prob}%"
        # annotate frame with model prediction
        annotated_frame = annotate_frame(original_frame, prediction_text, "green", position="top_right")
        output_writer.write(annotated_frame)

    print(f"Evaluation of video took {time() - start_time} seconds")


def main():
    """
    Main function that accepts input video, output video and train pickle file paths from the command line
        and then sequentially runs expert annotation and model prediction on the input video file

    Raises:
        FileNotFoundError: If input video file or training pickle file is not found
        ValueError: If input or output video file is not in mkv format
    """
    print(efn.__file__)  # so that reformat code does not remove efn import
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_video", help="Input video path", required=True)
    parser.add_argument("-o", "--output_video", help="Output video path", required=True)
    parser.add_argument("-t", "--train_pickle", help="Training pickle file path", required=True)
    args = vars(parser.parse_args())

    existing_file_parser_keys = ["input_video", "train_pickle"]
    for key in existing_file_parser_keys:
        if not os.path.isfile(args[key]):
            raise FileNotFoundError(f"File '{args[key]}' not found!")

    input_video_path = os.path.abspath(args["input_video"])
    train_pickle_path = os.path.abspath(args["train_pickle"])
    output_video_path = os.path.abspath(args["output_video"])

    if not input_video_path.endswith("mkv") or not output_video_path.endswith("mkv"):
        raise ValueError("Only mkv files supported by the program!")

    input_vid_dir = os.path.dirname(input_video_path)
    timestamp_str = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    evaluated_interim_path = os.path.join(input_vid_dir, f"evaluated_{timestamp_str}.mkv")

    """
    Steps:
    1. Annotate (top right, green) video with model prediction and prediction probability to generate 
    an interim video (evaluated_interim_path) 
    2. Get timestamps and annotation text labels corresponding to timestamps from the expert using program inputs,
    by calling the cli_annotation_input() function on the original input video.
    3. Annotate (top left, yellow) the interim video generated in step 1 with text labels provided by experts 
    accompanied by the timestamp ranges in which each text label is to appear.
    """

    evaluate_video(input_video_path, evaluated_interim_path, train_pickle_path)  # Step 1
    timestamps, text_list = cli_annotation_input(input_video_path)  # Step 2
    annotate_video(evaluated_interim_path, timestamps, text_list, output_video_path)  # Step 3
    Path(evaluated_interim_path).unlink(missing_ok=True)  # interim video is annotated with only the model output


if __name__ == "__main__":
    main()
