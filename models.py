from keras.models import Model, Sequential, load_model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Conv2D, MaxPooling2D, \
    Conv2DTranspose, Dropout, Flatten, Dense, ZeroPadding3D, Cropping3D
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.optimizers import Adam
import keras
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from math import log10, floor
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
from data_generators import *
from keras import regularizers
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff, cdist
from keras.utils import np_utils

image_size = [144, 144, 144]

def dice_loss(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices =  tf.math.divide(2 * intersection, card_y_true + card_y_pred)
    return -tf.reduce_mean(dices)

def dice_loss_smooth(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true, [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred, [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices =  tf.math.divide(2 * intersection+1, card_y_true + card_y_pred+1)
    return -tf.reduce_mean(dices)

def unet_3d(params):
    nb_layers = len(params['feat_maps'])

    if params['modality'] in ['pet', 'ct']:
        n_input_channels = 1
    elif params['modality']=='dual':
        n_input_channels = 2
    else:
        print('ERROR: Unknown modality.')

    # Input layer
    inputs = Input(batch_shape=(None, *image_size, n_input_channels))

    # Encoding part
    skips = []
    x = inputs
    for block_num in range(nb_layers-1):
        nb_features = params['feat_maps'][block_num]
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Bottleneck
    nb_features = params['feat_maps'][-1]
    x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)

    # Decoding part
    for block_num in reversed(range(nb_layers-1)):
        nb_features = params['feat_maps'][block_num]
        x = concatenate([Conv3DTranspose(nb_features, (2, 2, 2), strides=(2, 2, 2), padding='same')(x),
                         skips[block_num]], axis=4)
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(x)

    print('outputs.shape.dims', outputs.shape.dims)

    model = Model(inputs=[inputs], outputs=[outputs])

    if params['loss']=='dice_loss':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss)
    elif params['loss']=='dice_loss_smooth':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss_smooth)
    return model

def unet_3d_pad(params):
    inputs = Input(batch_shape=(None, *image_size, 1))


    conv1 = Conv3D(params['feat_maps'][0], (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(params['feat_maps'][0], (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(params['feat_maps'][1], (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(params['feat_maps'][1] * 2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(params['feat_maps'][2], (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(params['feat_maps'][2], (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(params['feat_maps'][3], (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(params['feat_maps'][3], (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(params['feat_maps'][4], (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['feat_maps'][4], (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = ZeroPadding3D(padding=((0,1),(0,1),(0,1)))(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    conv6 = Conv3D(params['feat_maps'][5], (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['feat_maps'][5], (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate(
        [Conv3DTranspose(params['feat_maps'][4], (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv5],axis=4)
    up7 = Cropping3D(cropping=((0,1),(0,1),(0,1)))(up7)
    conv7 = Conv3D(params['feat_maps'][4], (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['feat_maps'][4], (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate(
        [Conv3DTranspose(params['feat_maps'][3], (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv4],
        axis=4)
    conv8 = Conv3D(params['feat_maps'][3], (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['feat_maps'][3], (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate(
        [Conv3DTranspose(params['feat_maps'][2], (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv3],
        axis=4)
    conv9 = Conv3D(params['feat_maps'][2], (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['feat_maps'][2], (3, 3, 3), activation='relu', padding='same')(conv9)

    up10 = concatenate(
        [Conv3DTranspose(params['feat_maps'][1], (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), conv2],
        axis=4)
    conv10 = Conv3D(params['feat_maps'][1], (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['feat_maps'][1], (3, 3, 3), activation='relu', padding='same')(conv10)

    up11 = concatenate(
        [Conv3DTranspose(params['feat_maps'][0], (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), conv1],
        axis=4)
    conv11 = Conv3D(params['feat_maps'][0], (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['feat_maps'][0], (3, 3, 3), activation='relu', padding='same')(conv11)

    # Output layer
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv11)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(params['lr']), loss=dice_loss_smooth)
    return model





