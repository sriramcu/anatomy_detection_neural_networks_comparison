"""
Module used by both model-based and expert-based framewise class prediction to annotate\
    or write said prediction as a text label onto video frame with any given position or color.
"""
import os
from pathlib import Path
import sys
import cv2
import argparse

import numpy as np
from pymediainfo import MediaInfo
from time import time

from constants import *


def annotate_frame(frame, text, color_str, position="top_left"):
    """
    Annotates a frame with a text label with a given color and position

    Args:
        frame (np.ndarray): Image or video frame to be annotated
        text (str): Text to be displayed on annotation
        color_str (str): Color of the text to be displayed
        position (str, optional): Position of text label wrt frame (top right or left) . Defaults to "top_left".

    Raises:
        ValueError: If position parameter passed to the function is neither top right nor top left

    Returns:
        np.ndarray: Annotated frame
    """

    # Returns new, modified frame, not in-place
    
    height, width, _ = frame.shape
    
    pixels_needed  = len(text) * int(230/14 + 1) + 5

    
    if position == "top_right":   
        x = width - pixels_needed
        y = 40
        
    elif position == "top_left":
        x = 10
        y = 160  # Existing kvasir annotation occupies this place
    
    else:
        raise ValueError(f"Invalid frame position '{position}'")   
    
    x = int(x)
    y = int(y)
    
    modified_frame = frame.copy()    
    
    font_size = 0.9
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    
    # BGR for cv2
    color_dict = COLOR_DICT
    
    font_color = color_dict[color_str]
    
    modified_frame = cv2.putText(modified_frame, 
                                text, 
                                (x,y), 
                                font, 
                                font_size, 
                                font_color, 
                                font_thickness)  
    
    return modified_frame


def annotate_video(input_video_path, timestamps, text_list, output_video_path):
    """
    Manually annotates videos by calling the annotate_frame() function framewise. Used by experts.

    Args:
        input_video_path (str): Path to input video
        timestamps (list): list of lists of timestamp ranges; each sublist corresponds to a label given by text_list
        text_list (list): list of text labels to be annotated in each timestamp range given by timestamps list
        output_video_path (str): Path to output annotated video
    """
        
    start_time = time()

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    success, frame = vidcap.read()
    count = 0
    
    timestamp_idx = 0                                                   
    
    Path(output_video_path).unlink(missing_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))            
    
    while success:
        success, frame = vidcap.read()
        
        if not success:
            break
        
        count+=1
        lower_timestamp = timestamps[timestamp_idx][0]
        upper_timestamp = timestamps[timestamp_idx][1]
        
        current_ts = count/fps
        # calculates current timestamp of frame being processed; if between lower_timestamp and upper_timestamp,
        # then annotate that frame corresponding to the current timestamp range by using the same index variable 
        # in the timestamps list and the text_list

        if current_ts >= lower_timestamp and current_ts <= upper_timestamp:
            modified_frame = annotate_frame(frame, text_list[timestamp_idx], "yellow")
            # top left position (default)
            output_writer.write(modified_frame)
            # write modified frame to output video path specified by function argument
        
        elif current_ts > upper_timestamp:
            # go to the next index position of both timestamps list and text_list
            timestamp_idx += 1
            if timestamp_idx >= len(timestamps):
                break
            
    print(f"Annotation of video took {time()-start_time} seconds")



def verify_timestamps(timestamps, vid_duration):
    """
    Checks if list of lists of timestamps ranges are continuous and in order

    Args:
        timestamps (list): list of lists of timestamps ranges
        vid_duration (int): duration of video in seconds

    Raises:
        ValueError: if sublists i.e. timestamp ranges are not in ascending order
        ValueError: if max timestamp value is greater than the video duration or if min value is negative
    """
    flattened_timestamps = list(np.array(timestamps).flatten())
    
    if sorted(flattened_timestamps) != flattened_timestamps:
        raise ValueError("Timestamps entered non-sequentially.")
    
    if max(flattened_timestamps) > vid_duration or min(flattened_timestamps) < 0:
        raise ValueError("All timestamps must be in seconds between 0 and video duration.")
        
        

def cli_annotation_input(input_video_path):
    """
    Interactive function for manual annotation of video by expert

    Args:
        input_video_path (str): path to input video

    Returns:
        list: list of lists of timestamps ranges
        list: list of text labels to be annotated in each timestamp range given by timestamps list
    """
    timestamps = []
    text_list = []
    
    media_info = MediaInfo.parse(input_video_path)
    #duration in milliseconds
    duration_in_ms = media_info.tracks[0].duration
    
    vid_duration = int(duration_in_ms/1000)        
     
    print(f"Video duration = {vid_duration} seconds")       
    
    while True:
        start_ts = int(input("Enter starting timestamp in seconds (-1 for stopping): "))
        if start_ts == -1:
            break
        
        end_ts = int(input("Enter ending timestamp in seconds (-1 for end of video): "))
        
        if end_ts == -1:            
            end_ts = vid_duration
        
        text = input("Enter text to be annotated: ")
        timestamps.append([start_ts, end_ts])
        text_list.append(text)
    
    verify_timestamps(timestamps, vid_duration)    
     
    print(timestamps, text_list)   
    
    return timestamps, text_list                
        
        

def main():
    """
    Main function run to test only manual expert annotation- \
        second function gets timestamps and text list based on user input\
        which are passed to the second function to actually annotate the input video 
    """
    input_video_path = os.path.abspath(sys.argv[1])
    output_video_path = os.path.abspath(sys.argv[2])
    
    timestamps, text_list = cli_annotation_input(input_video_path) 

    annotate_video(input_video_path, timestamps, text_list, output_video_path)
    
    
if __name__ == "__main__":
    main()