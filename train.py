"""
Standalone program to train the model using command-line arguments 
to select the network name and some hyperparameters not mentioned in constants module
"""
import argparse
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import math
from collections import Counter
import numpy as np
import faulthandler

from utils.constants import *
from utils.my_datagen import return_my_datagen
from utils.neural_networks import *

faulthandler.enable()


def train(
    num_classes,
    network_name, 
    num_epochs,
    train_dir, 
    val_dir,    
    custom_preprocessing
    ):   
    """
    Function to train neural network based on parameters stored in constants.py \
    as well as those supplied via the function parameters. Stores training metrics\
    like accuracy, loss and hyperparameters used in a pickle file

    Args:
        num_classes (int): number of classes predicted by neural network        
        network_name (str): name of neural network used for training
        num_epochs (int): number of epochs to train the model for
        train_dir (str): train dataset directory
        val_dir (str): validation dataset directory
        custom_preprocessing (bool): specifies whether to use custom preprocessing for model training

    Raises:
        FileNotFoundError: if training dataset directory is not found
        FileNotFoundError: if validation dataset directory is not found
        ValueError: if neural network name doesn't match any of the supported networks

    Returns:
        None: No return value. Function trains the model; saves parameters, h5 file & train history
    """

    checkpoint_dir = CHECKPOINTS_DIR
    
    checkpoint_filename = f"{network_name}_{num_epochs}epochs.h5"

    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)

    parameters_filepath = os.path.join(TEXT_FILES_DIR, "parameters.txt")

    metrics_pickle_filename = f"train_metrics_{num_epochs}epochs_{network_name}.pickle"

    metrics_pickle_filepath = os.path.join(TRAIN_PICKLE_DIR, 
                                           metrics_pickle_filename)    

    if not os.path.isdir(train_dir):
        raise FileNotFoundError("Training directory not found!")
    
    if not os.path.isdir(val_dir):
        raise FileNotFoundError("Validation directory not found!")    
    
    train_datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="training")
        
    val_datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="prediction")

    train_generator = train_datagen.flow_from_directory(
                                                  train_dir,
                                                  target_size=(TRAIN_IMAGE_WIDTH,TRAIN_IMAGE_HEIGHT),
                                                  class_mode="categorical", 
                                                  batch_size=BATCH_SIZE
                                                  )    
    
    val_gen = val_datagen.flow_from_directory(
                                          val_dir,
                                          target_size=(TRAIN_IMAGE_WIDTH,TRAIN_IMAGE_HEIGHT),
                                          class_mode="categorical", 
                                          batch_size=BATCH_SIZE                                          
                                          )      
    
    
    if network_name == "inceptionv4":
        model = create_inception_v4(num_classes)        

    elif network_name == "inceptionv3":
        model = create_pretrained_inceptionv3(num_classes)

        
    elif network_name == "efficientnet" or network_name == "efficientnetb7":
        model = create_pretrained_efficientnetb7(num_classes)   
    
    elif network_name == "nasnet":
        model = create_pretrained_nasnet(num_classes)
    
    else:
        raise ValueError("Please check your network name and try again")    
    
    def lr_scheduler(epoch, lr):
        if epoch > (num_epochs/3):
            lr = LEARNING_RATE/10
        
        # print("Learning Rate = ", lr)
        return lr    
    
    
    mc = keras.callbacks.ModelCheckpoint(
                                        checkpoint_filepath,
                                        save_weights_only=False,  
                                        monitor='val_accuracy',                                        
                                        save_best_only=True
                                        )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor=f"val_loss",
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    )
    
    lr_epoch_based_callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    lr_val_acc_based_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=LR_DECAY_FACTOR,
        patience=20,
        min_lr=LEARNING_RATE * LR_DECAY_FACTOR,
    )


    # Earlier option to use RMSProp has been removed, as results were unsatisfactory

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer= optimizer,
                  metrics=["accuracy"])

    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 

    history = model.fit(train_generator,
                     epochs=num_epochs,
                     verbose=True,
                     validation_data=val_gen,
                     callbacks=[mc, early_stopping_callback, lr_val_acc_based_callback], 
                     class_weight=class_weights)    
    
    
    f = open(parameters_filepath, 'a')
    params = [num_epochs, MY_DROPOUT, LEARNING_RATE, L2_REG, custom_preprocessing, network_name]
    f.write(f"{params}\n")
    f.close()
    
    
    f = open(metrics_pickle_filepath, "wb")
    # Pickle dumps are FIFO
    pickle.dump(history.history, f)
    params.append(checkpoint_filename) 
    pickle.dump(params, f) 
    f.close()
        

def main():
    """
    Calls train() based on command line arguments and values stored in constants.py
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    
    # Default arguments
    parser.add_argument("-n", "--network", default="inceptionv3", type=str, help="Name of the network")
        
    parser.add_argument("-train", 
                        "--train_dir", 
                        type=str,
                        default=TRAIN_DIR, 
                        help="train dataset directory")
    
    parser.add_argument("-val", "--val_dir", type=str, default=VAL_DIR, help="val dataset directory")
        
    requiredNamed = parser.add_argument_group('required named arguments')
    # Required arguments
    requiredNamed.add_argument("-c", "--num_classes", type=int, required=True, help=" ")
    
    requiredNamed.add_argument("-p", "--custom_preprocess", type=int, choices=[0,1], required=True, help=" ")
    
    requiredNamed.add_argument("-e", "--epochs", type=int, required=True, help=" ")
    
    args = vars(parser.parse_args())

    num_classes = int(args["num_classes"])
    
    custom_preprocessing = int(args["custom_preprocess"])
    
    train_dir = str(args['train_dir'])

    val_dir = str(args['val_dir'])    
    
    num_epochs = int(args['epochs'])
    
    network_name = args["network"]

    train(num_classes, network_name, num_epochs, train_dir, val_dir, custom_preprocessing)


if __name__ == "__main__":
    main()