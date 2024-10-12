"""
Module used to evaluate a test video by performing model predictions on it
"""
import os
import pickle
import sys
from pathlib import Path
from time import time

import cv2
from keras.models import load_model

from generate_datagen import get_datagen_obj
from predict_frame import test_frame
from utils.constants import CHECKPOINTS_DIR, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, CLASS_LABELS
from video_annotator import annotate_frame


def evaluate_video(input_video_path, output_video_path, metrics_pickle_filepath):
    """
    Evaluates an input video by calling the predict_frame.test_frame() function framewise and annotating it\
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

    model = load_model(checkpoints_filepath)

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

        class_names = CLASS_LABELS

        if count == 1:
            print("Loading CUDNN for model prediction...")

        _, predicted_class, rounded_prob = test_frame(frame, model,
                                                      datagen, class_names)
        if count == 1:
            print("CUDNN loaded for model prediction, first frame predicted!")

        elif count % 250 == 0:
            print(f"{count} frames of the test video have been predicted so far...")

            # run model prediction on current frame and extract the prediction result for annotation purposes

        pred_txt = f"{predicted_class}:{rounded_prob}%"

        # annotate frame with model prediction
        annotated_frame = annotate_frame(original_frame, pred_txt, "green", position="top_right")

        output_writer.write(annotated_frame)

    print(f"Evaluation of video took {time() - start_time} seconds")


if __name__ == "__main__":
    evaluate_video(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]),
                   os.path.abspath(sys.argv[3]))
