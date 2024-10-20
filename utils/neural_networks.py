"""
Module that contains architecture of all neural networks used in this project
"""
import os

from utils.constants import DISABLE_TF_LOGGING, TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, MY_DROPOUT, NUM_TRAINABLE_LAYERS, \
    L2_REG

if DISABLE_TF_LOGGING:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# To disable tensorflow debugging log messages 

from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.layers import concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers
from tensorflow.keras.applications.nasnet import NASNetLarge
import efficientnet.keras as efn


def conv_block(x, nb_filter, nb_row, nb_col, padding='same', subsample=(1, 1), bias=False):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Conv2D(nb_filter, (nb_row, nb_col), strides=subsample, padding=padding, use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def inception_stem(input):
    channel_axis = -1

    # Shape 299 x 299 x 3 
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), padding='valid')
    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding='valid')
    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), padding='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = concatenate([x1, x2], axis=channel_axis)

    return x


def inception_a(input):
    channel_axis = -1

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = concatenate([a1, a2, a3, a4], axis=channel_axis)

    return m


def inception_b(input):
    channel_axis = -1

    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = concatenate([b1, b2, b3, b4], axis=channel_axis)

    return m


def inception_c(input):
    channel_axis = -1

    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)

    c2 = concatenate([c2_1, c2_2], axis=channel_axis)

    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)

    c3 = concatenate([c3_1, c3_2], axis=channel_axis)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = concatenate([c1, c2, c3, c4], axis=channel_axis)

    return m


def reduction_a(input):
    channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), padding='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)

    return m


def reduction_b(input):
    channel_axis = -1

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), padding='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)

    return m


def create_inception_v4(nb_classes):
    """
    Creates Inceptionv4 model based on previously defined sub-architectures like inception_stem etc

    Args:
        nb_classes (int): number of classes predicted by the model

    Returns:
        keras.Model: model generated by the function
    """

    print("Custom Inceptionv4 model being used...")
    init = Input((TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, 3))
    x = inception_stem(init)

    # 4 x Inception A
    for _ in range(4):
        x = inception_a(x)

    # Reduction A
    x = reduction_a(x)

    # 7 x Inception B
    for _ in range(7):
        x = inception_b(x)

    # Reduction B
    x = reduction_b(x)

    # 3 x Inception C
    for _ in range(3):
        x = inception_c(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)
    # Dropout 
    x = Dropout(MY_DROPOUT)(x)
    x = Flatten()(x)
    # Output
    out = Dense(nb_classes, activation='softmax')(x)
    model = Model(init, out, name='Inception-v4')
    return model


def create_pretrained_inceptionv3(nb_classes):
    """
    Creates Inceptionv3 pretrained model

    Args:
        nb_classes (int): number of classes predicted by the model

    Returns:
        keras.Model: model generated by the function
    """
    print("Pretrained Inceptionv3 being used...")
    base_model = InceptionV3(input_shape=(TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, 3), include_top=False,
                             weights='imagenet')
    l = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
        rev_index = l - i - 1
        if rev_index < NUM_TRAINABLE_LAYERS:
            layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(30, activation="relu",
              kernel_regularizer=regularizers.l2(L2_REG),
              activity_regularizer=regularizers.l2(L2_REG),
              bias_regularizer=regularizers.l2(L2_REG)
              )(x)
    x = Dropout(MY_DROPOUT)(x)
    x_image_out = Dense(nb_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        activity_regularizer=regularizers.l2(L2_REG),
                        bias_regularizer=regularizers.l2(L2_REG)
                        )(x)
    model = Model(base_model.input, x_image_out)
    return model


def create_pretrained_efficientnetb7(nb_classes):
    """
    Creates EfficientNetB7 pretrained model

    Args:
        nb_classes (int): number of classes predicted by the model

    Returns:
        keras.Model: model generated by the function
    """

    print("Pretrained EfficientNet B7 being used...")
    base_model = efn.EfficientNetB7(input_shape=(TRAIN_IMAGE_WIDTH, TRAIN_IMAGE_HEIGHT, 3),
                                    include_top=False, weights='imagenet')
    l = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
        rev_index = l - i - 1
        if rev_index < NUM_TRAINABLE_LAYERS:
            layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(30, activation="relu",
              kernel_regularizer=regularizers.l2(L2_REG),
              activity_regularizer=regularizers.l2(L2_REG),
              bias_regularizer=regularizers.l2(L2_REG)
              )(x)
    x = Dropout(MY_DROPOUT)(x)
    x_image_out = Dense(nb_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        activity_regularizer=regularizers.l2(L2_REG),
                        bias_regularizer=regularizers.l2(L2_REG)
                        )(x)
    model = Model(base_model.input, x_image_out)
    return model


def create_pretrained_nasnet(nb_classes):
    """
    Creates nasnet pretrained model

    Args:
        nb_classes (int): number of classes predicted by the model

    Returns:
        keras.Model: model generated by the function
    """

    print("Pretrained NASNet being used...")
    base_model = NASNetLarge(input_shape=(331, 331, 3), include_top=False, weights='imagenet')
    l = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
        rev_index = l - i - 1
        if rev_index < NUM_TRAINABLE_LAYERS:
            layer.trainable = True
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(30, activation="relu",
              kernel_regularizer=regularizers.l2(L2_REG),
              activity_regularizer=regularizers.l2(L2_REG),
              bias_regularizer=regularizers.l2(L2_REG)
              )(x)
    x = Dropout(MY_DROPOUT)(x)
    x_image_out = Dense(nb_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        activity_regularizer=regularizers.l2(L2_REG),
                        bias_regularizer=regularizers.l2(L2_REG)
                        )(x)
    model = Model(base_model.input, x_image_out)
    return model
