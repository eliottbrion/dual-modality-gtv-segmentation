# Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# Ce purpose of these data generators are: (i) load images one by one in the RAM and (ii) perform online 3D data augmentation

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
# from utils import *
import random
import pickle
from scipy import ndimage

image_size = [144, 144, 144]
spacing = [1., 1., 1.]
data_dir = '/DATA/public/hecktor_data'


class DataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, modality, params):
        'Initialization'
        self.dim = tuple(image_size)
        self.batch_size = params['batch_size']
        self.list_IDs = list_IDs
        self.modality = modality
        if modality in ['pet', 'ct']:
            self.n_channels = 1
        elif modality=='dual':
            self.n_channels = 2
        else:
            print('ERROR: Unknown modality.')
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # n_organs = len(organs_names)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            shears = np.array([0.02 * random.uniform(-1, 1) for _ in range(6)])
            angles = np.array([5 * random.uniform(-1, 1) for _ in range(3)])
            shifts = np.array([0.05 * random.uniform(-1, 1) * image_size[i] for i in range(3)])

            if self.modality in ['pet','dual']:
                pet = np.load(data_dir + '/' + ID + '/pt.npy')
                pet = np.clip(pet, 0, 10)
                vmin = np.min(pet)
                vmax = np.max(pet)
                pet = (pet-vmin)/(vmax-vmin)
                pet = image_transform(pet, shears, angles, shifts, order=3)
            if self.modality in ['ct','dual']:
                ct = np.load(data_dir + '/' + ID + '/ct.npy')
                ct = np.clip(ct, -200, 200)
                vmin = np.min(ct)
                vmax = np.max(ct)
                ct = (ct - vmin) / (vmax - vmin)
                ct = image_transform(ct, shears, angles, shifts, order=3)

            if self.modality=='pet':
                X[i,] = np.expand_dims(pet, axis=-1)
            elif self.modality=='ct':
                X[i,] = np.expand_dims(ct, axis=-1)
            elif self.modality=='dual':
                X[i, :, :, :, 0] = pet
                X[i, :, :, :, 1] = ct
            else:
                print('ERROR: Unknown modality.')

            # Store class
            masks = np.load(data_dir + '/' + ID + '/gtv.npy')
            masks_trans = image_transform(masks, shears, angles, shifts,order=0)
            Y[i,:,:,:,0] = masks_trans

        return X, Y


class DataGeneratorVal(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, modality, params):
        'Initialization'
        self.dim = tuple(image_size)
        self.batch_size = params['batch_size']
        self.list_IDs = list_IDs
        self.n_channels = 1
        self.shuffle = False
        if modality in ['pet', 'ct']:
            self.n_channels = 1
        elif modality == 'dual':
            self.n_channels = 2
        else:
            print('ERROR: Unknown modality.')
        self.modality = modality
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        # n_organs = len(organs_names)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.modality in ['pet', 'dual']:
                pet = np.load(data_dir + '/' + ID + '/pt.npy')
                pet = np.clip(pet, 0, 10)
                vmin = np.min(pet)
                vmax = np.max(pet)
                pet = (pet - vmin) / (vmax - vmin)
            if self.modality in ['ct', 'dual']:
                ct = np.load(data_dir + '/' + ID + '/ct.npy')
                ct = np.clip(ct, -200, 200)
                vmin = np.min(ct)
                vmax = np.max(ct)
                ct = (ct - vmin) / (vmax - vmin)

            if self.modality == 'pet':
                X[i,] = np.expand_dims(pet, axis=-1)
            elif self.modality == 'ct':
                X[i,] = np.expand_dims(ct, axis=-1)
            elif self.modality == 'dual':
                X[i, :, :, :, 0] = pet
                X[i, :, :, :, 1] = ct
            else:
                print('ERROR: Unknown modality.')

            # Store class
            # Y[i,] = np.load('data/' + ID + '-mask.npy')
            Y[i,:,:,:,0] = np.load(data_dir + '/' + ID + '/gtv.npy')

        return X, Y


def image_transform(image, shears, angles, shifts, order):
    shear_matrix = np.array([[1, shears[0], shears[1], 0],
                             [shears[2], 1, shears[3], 0],
                             [shears[4], shears[5], 1, 0],
                             [0, 0, 0, 1]])

    shift_matrix = np.array([[1, 0, 0, shifts[0]],
                             [0, 1, 0, shifts[1]],
                             [0, 0, 1, shifts[2]],
                             [0, 0, 0, 1]])

    offset = np.array([[1, 0, 0, int(image_size[0] / 2)],
                       [0, 1, 0, int(image_size[1] / 2)],
                       [0, 0, 1, int(image_size[2] / 2)],
                       [0, 0, 0, 1]])

    offset_opp = np.array([[1, 0, 0, -int(image_size[0] / 2)],
                           [0, 1, 0, -int(image_size[1] / 2)],
                           [0, 0, 1, -int(image_size[2] / 2)],
                           [0, 0, 0, 1]])

    angles = np.deg2rad(angles)
    rotx = np.array([[1, 0, 0, 0],
                     [0, np.cos(angles[0]), -np.sin(angles[0]), 0],
                     [0, np.sin(angles[0]), np.cos(angles[0]), 0],
                     [0, 0, 0, 1]])
    roty = np.array([[np.cos(angles[1]), 0, np.sin(angles[1]), 0],
                     [0, 1, 0, 0],
                     [-np.sin(angles[1]), 0, np.cos(angles[1]), 0],
                     [0, 0, 0, 1]])
    rotz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, 0],
                     [np.sin(angles[2]), np.cos(angles[2]), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    rotation_matrix = offset_opp.dot(rotz).dot(roty).dot(rotx).dot(offset)
    affine_matrix = shift_matrix.dot(rotation_matrix).dot(shear_matrix)
    return ndimage.interpolation.affine_transform(image, affine_matrix, order=order, mode='nearest')