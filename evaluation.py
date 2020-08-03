from keras.models import load_model
import numpy as np
import os
from scipy import ndimage
import pickle
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff, cdist
from models import *
from utils import *

image_size = [144, 144, 144]
data_dir = '/DATA/public/hecktor_data'

def evaluate(filenames, src_dir, gpu, modality, save_predictions=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    n_images = len(filenames)
    print('loading model...')
    model = load_model(src_dir + '/weights.h5', custom_objects={'dice_loss': dice_loss})
    print('done.')

    print('Progress:')
    n_patients = len(filenames)
    for patient_num in range(n_patients):
        filename = filenames[patient_num]
        if os.path.exists(src_dir + '/predictions/' + filename + '_prediction.npy'):
            # Load previously saved prediction
            prediction = np.load(src_dir + '/predictions/' + filename + '_prediction.npy')
        else:
            if modality in ['pet', 'dual']:
                pet = np.load(data_dir + '/' + filename + '/pt.npy')
                pet = np.expand_dims(pet, axis=0)
                # pet = np.expand_dims(pet, axis=-1)
                pet = np.clip(pet, 0, 10)
                vmin = np.min(pet)
                vmax = np.max(pet)
                pet = (pet - vmin) / (vmax - vmin)
            if modality in ['ct', 'dual']:
                ct = np.load(data_dir + '/' + filename + '/ct.npy')
                ct = np.expand_dims(ct, axis=0)
                # ct = np.expand_dims(ct, axis=-1)
                ct = np.clip(ct, -200, 200)
                vmin = np.min(ct)
                vmax = np.max(ct)
                ct = (ct - vmin) / (vmax - vmin)

            if modality == 'pet':
                X = np.expand_dims(pet, axis=-1)
            elif modality == 'ct':
                X = np.expand_dims(ct, axis=-1)
            elif modality == 'dual':
                X = np.zeros((1,*image_size,2))
                X[0, :, :, :, 0] = pet
                X[0, :, :, :, 1] = ct
            else:
                print('ERROR: Unknown modality.')

            prediction = model.predict(X)
            prediction = prediction[0, :, :, :, 0]
            del X
            if save_predictions:
                np.save(src_dir + '/predictions/' + filename + '_prediction.npy', prediction)

        prediction = (prediction>0.5)
        mask = np.load(data_dir + '/' + filename + '/gtv.npy')
        metrics = {'DSC': f1_score(mask.flatten(), prediction.flatten())}
        if not os.path.exists(src_dir + '/metrics'):
            os.makedirs(src_dir + '/metrics')
        pickle.dump(metrics, open(src_dir + '/metrics/' + filename + '_metrics.p', "wb"))
        print('|-- ' + str(patient_num) + ' ' + str(metrics['DSC']))



