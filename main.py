"""
Standalone program that sequentially runs expert annotation and model prediction on a video file
"""
import argparse
import datetime
import os
from pathlib import Path

from utils.video_annotator import cli_annotation_input, annotate_video
from utils.video_evaluate import evaluate_video


def main():
    """
    Main function that accepts input video, output video and train pickle file paths from the command line\
        and then sequentially runs expert annotation and model prediction on the input video file

    Raises:
        FileNotFoundError: If input video file or training pickle file is not found
        ValueError: If input or output video file is not in mkv format
    """

    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", help="Input video path", required=True)
    parser.add_argument("-o", help="Output video path", required=True)
    parser.add_argument("-tp", help="Training pickle file path", required=True)

    args = vars(parser.parse_args())

    existing_file_parser_keys = ["i", "tp"]

    for key in existing_file_parser_keys:
        if not os.path.isfile(args[key]):
            raise FileNotFoundError(f"File '{args[key]}' not found!")

    input_video_path = os.path.abspath(args["i"])
    train_pickle_path = os.path.abspath(args["tp"])
    output_video_path = os.path.abspath(args["o"])

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

    Path(evaluated_interim_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
