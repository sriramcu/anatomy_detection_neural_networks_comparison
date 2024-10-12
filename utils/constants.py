"""
Module to store commonly used hyperparameters, directories and other variables to provide an easy 
way to modify commonly changed values to tune model performance
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # root directory of the grip assessment project
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")  # dataset directory for grip assessment
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TRAIN_PICKLE_DIR = os.path.join(ROOT_DIR, "training_metrics_pickle_files")
CONF_PICKLE_DIR = os.path.join(ROOT_DIR, "confusion_matrix_pickle_files")
CONF_CSV_DIR = os.path.join(ROOT_DIR, "confusion_matrix_tables")
GRAPHS_ROOT_DIR = os.path.join(ROOT_DIR, "graphs")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")  # trained h5 files stored here
TEXT_FILES_DIR = os.path.join(ROOT_DIR ,"text_files")  
# text files to store analysis, parameters used, comments, etc.
INPUT_VIDEOS_DIR = os.path.join(ROOT_DIR, "videos", "input_videos")
OUTPUT_VIDEOS_DIR = os.path.join(ROOT_DIR, "videos", "output_videos")
# directory to store output annotated videos

DISABLE_TF_LOGGING = True
# controls whether to disable tensorflow debug logging like information (I) logs

TRAIN_IMAGE_WIDTH = 299
TRAIN_IMAGE_HEIGHT = 299
# input layer dimensions of neural network. Even if above values do not match neural network actually used 
# at the time of execution, program has a provision to prompt the user for actual dims in case of mismatch


NUM_TRAINABLE_LAYERS = 0
L2_REG = 0.005
MY_DROPOUT = 0.2
LEARNING_RATE = 0.001
# for pretrained networks, number of layers whose parameters/weights can be modified during training process

BATCH_SIZE = 32

SAMPLEWISE_NORM = False
# controls whether Keras ImageDataGenerator object instantiated will use sample wise standard normalisation



CLASS_LABELS = ["cecum","esophagitis", "polyps", "pylorus","ulcerative-colitis","z-line"]

LR_DECAY_FACTOR = 0.1
# factor by which learning rate decays after gains in validation accuracy slows for a certain no. of epochs

COLOR_DICT = {
              "red": (0,0,255), 
              "dark_blue": (255,0,0),
              "blue": (200,30,0), 
              "green":(0,255,0), 
              "yellow":(0,255,255),
              "orange":(0,140,255)              
              }
# Dictionary defining color BGR values (str-tuple pairs)