"""
Standalone program to train the model using command-line arguments 
to select the network name and some hyperparameters not mentioned in constants module
"""
import argparse
import faulthandler
import os
import pickle
from collections import Counter

import keras
import tensorflow as tf

from utils.constants import BATCH_SIZE, TRAIN_IMAGE_HEIGHT, TRAIN_IMAGE_WIDTH, LEARNING_RATE, LR_DECAY_FACTOR, \
    MY_DROPOUT, L2_REG, TRAIN_DIR, CHECKPOINTS_DIR, TRAIN_PICKLE_DIR, VAL_DIR, PARAMETERS_FILEPATH
from utils.generate_datagen import get_datagen_obj
from utils.neural_networks import create_inception_v4, create_pretrained_inceptionv3, create_pretrained_efficientnetb7, \
    create_pretrained_nasnet

faulthandler.enable()


def train(
        num_classes,
        network_name,
        num_epochs,
        train_dir,
        val_dir,
        custom_preprocessing,
        custom_name
):
    """
    Function to train neural network based on parameters stored in constants.py (containing directories for datasets,
    checkpoint files, train pickle files, text files, train img dimensions, LR + decay, batch size, L2, dropout)
    as well as the function parameters.
    Stores hyperparameters into parameters.txt. These, in addition to the training metrics like accuracy, loss and the
    checkpoint filename, are also stored into a pickle file whose directory is mentioned in the constants module.

    Args:
        num_classes (int): number of classes predicted by neural network        
        network_name (str): name of neural network used for training
        num_epochs (int): number of epochs to train the model for
        train_dir (str): train dataset directory
        val_dir (str): validation dataset directory
        custom_preprocessing (bool): specifies whether to use custom preprocessing for model training
        custom_name (str): custom string to distinguish model file name

    Raises:
        FileNotFoundError: if training dataset directory is not found
        FileNotFoundError: if validation dataset directory is not found
        ValueError: if neural network name doesn't match any of the supported networks

    Returns:
        None: No return value. Function trains the model; saves parameters, h5 file & train history
    """
    if custom_name != "" and not custom_name.startswith("_"):
        custom_name = "_" + custom_name
    checkpoint_filename = f"{network_name}{custom_name}_{num_epochs}epochs.h5"
    checkpoint_filepath = os.path.join(CHECKPOINTS_DIR, checkpoint_filename)
    metrics_pickle_filename = f"train_metrics_{num_epochs}epochs_{network_name}.pickle"
    metrics_pickle_filepath = os.path.join(TRAIN_PICKLE_DIR,
                                           metrics_pickle_filename)

    if not os.path.isdir(train_dir):
        raise FileNotFoundError("Training directory not found!")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError("Validation directory not found!")

    train_datagen = get_datagen_obj(custom_preprocessing=custom_preprocessing, mode="training")
    val_datagen = get_datagen_obj(custom_preprocessing=custom_preprocessing, mode="prediction")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT),
        class_mode="categorical",
        batch_size=BATCH_SIZE
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT),
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

    # Define callbacks

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        save_best_only=True
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    def lr_scheduler(epoch, lr):
        if epoch > (num_epochs / 3):
            lr = LEARNING_RATE / 10
        return lr

    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    lr_val_acc_based_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=LR_DECAY_FACTOR,
        patience=20,
        min_lr=LEARNING_RATE * LR_DECAY_FACTOR,
    )

    # Define optimizer - The results using RMSProp were unsatisfactory

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy"])

    # Set class weights to handle imbalanced dataset
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

    history = model.fit(train_generator,
                        epochs=num_epochs,
                        verbose=True,
                        validation_data=val_gen,
                        callbacks=[model_checkpoint_callback, early_stopping_callback, lr_val_acc_based_callback],
                        class_weight=class_weights)

    # Save stuff into txt and pickle files as explained in the docstring
    f = open(PARAMETERS_FILEPATH, 'a')
    params = [num_epochs, MY_DROPOUT, LEARNING_RATE, L2_REG, custom_preprocessing, network_name]
    f.write(f"{params}\n")
    f.close()

    f = open(metrics_pickle_filepath, "wb")
    pickle.dump(history.history, f)
    params.append(checkpoint_filename)
    pickle.dump(params, f)
    f.close()


def main():
    """
    Calls train() based on command line arguments. Default values (dataset dirs) stored in constants.py
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--network", default="inceptionv3", type=str, help="Name of the network")
    parser.add_argument("-t",
                        "--train_dir",
                        type=str,
                        default=TRAIN_DIR,
                        help="train dataset directory")
    parser.add_argument("-v", "--val_dir", type=str, default=VAL_DIR, help="val dataset directory")
    parser.add_argument("-c", "--custom_name", type=str, default="",
                        help="Custom string to append to checkpoints filename")
    required_arg_group = parser.add_argument_group('required named arguments')
    required_arg_group.add_argument("-c", "--num_classes", type=int, required=True, help=" ")
    required_arg_group.add_argument("-p", "--custom_preprocess", type=int, choices=[0, 1], required=True, help=" ")
    required_arg_group.add_argument("-e", "--epochs", type=int, required=True, help=" ")
    args = vars(parser.parse_args())

    num_classes = int(args["num_classes"])
    custom_preprocessing = bool(int(args["custom_preprocess"]))
    train_dir = str(args['train_dir'])
    val_dir = str(args['val_dir'])
    num_epochs = int(args['epochs'])
    network_name = args["network"]
    custom_name = args["custom_name"]

    train(num_classes, network_name, num_epochs, train_dir, val_dir, custom_preprocessing, custom_name)


if __name__ == "__main__":
    main()
