from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Flatten, Dense
import tensorflow as tf
import numpy as np
import os
import pickle
from scipy import ndimage
from data_generators import *


from models import *
from utils import *

image_size = [144,144,144]
spacing = [1., 1., 1.]
normalization_params = None


def train(partition, previous_dir, gpu, dest_dir, fold_name, params):

    # Set gpu and seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from numpy.random import seed
    seed(params['seed'])
    import tensorflow as tf
    tf.random.set_seed(params['seed'])

    # Create folder and subfolder
    if not os.path.exists(dest_dir + '/' + params2name(params)):
        os.makedirs(dest_dir + '/' + params2name(params))
        pickle.dump(params, open(dest_dir + '/' + params2name(params) + '/params.p', "wb"))

    crossvalidation_dir = dest_dir + '/' + params2name(params) + '/fold_' + fold_name
    if not os.path.exists(crossvalidation_dir):
        os.makedirs(crossvalidation_dir)
        pickle.dump(params, open(crossvalidation_dir + '/params.p', "wb"))
        pickle.dump(partition, open(crossvalidation_dir + '/partition.p', "wb"))
        np.save(crossvalidation_dir + '/previous_dir.npy', previous_dir)

    # Create or load model
    if previous_dir == None:
        print('Starting training from scratch.')
        params_previous = {'epochs': 0}
        if params['model']=='unet_3d':
            model = unet_3d(params)
        if params['model']=='unet_3d_pad':
            model = unet_3d_pad(params)
        elif params['model']=='unet_3d_smooth':
            model = unet_3d_smooth(params)
        hist1 = None  # no history from previous model
    else:
        params_previous = pickle.load(open(previous_dir + '/params.p', "rb"))
        print('Resuming training of a model trained for ' + str(params_previous['epochs']) + ' epochs.')
        hist1 = pickle.load(open(previous_dir + '/history.p', "rb"))  # history from previous model
        co = {'dice_loss': dice_loss}
        model = load_model(previous_dir + '/weights.h5', custom_objects=co)

    # Train
    print('params', params)
    training_generator = DataGeneratorTrain(partition['train'], params['modality'], params)
    validation_generator = DataGeneratorVal(partition['val'], params['modality'], params)

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=len(partition['train'])//params['batch_size'],
                                  validation_steps=len(partition['val'])//params['batch_size'],
                                  verbose=1,
                                  epochs=params['epochs'] - params_previous['epochs'])

    model.save(crossvalidation_dir + '/weights.h5')

    # Merge and save histories
    hist2 = history.history  # history for the new epochs
    if hist1 == None:  # No previous model
        hist = hist2
    elif hist2 == {}:
        hist = hist1
    else:
        hist = {}
        for key in hist1.keys():
            hist[key] = hist1[key] + hist2[key]
    pickle.dump(hist, open(crossvalidation_dir + '/history.p', "wb"))
