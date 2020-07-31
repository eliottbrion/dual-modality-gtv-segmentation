from models import *
from data_generators import *
from training import *
# from evaluation import *

data_dir = '/DATA/public/hecktor_data'

patients_old = np.load(data_dir + '/patients.npy')
patients = {}
for center in ['CHGJ', 'CHMR', 'CHUM', 'CHUS']:
    patients[center] = [elem for elem in patients_old if elem.startswith(center)]

test_center = 'CHGJ'
partition = {'train': patients['CHMR']+patients['CHUM']+patients['CHUS'],
             'val': patients[test_center]}
print('partition', partition)
print('n_train', len(partition['train']))
print('n_val', len(partition['val']))
gpu = 0
fold_name = test_center

dest_dir = 'results'

params = {'model': 'unet_3d',
          'modality': 'dual',
          'epochs': 10,
          'lr': 1e-4,
          'batch_size': 2,
          'feat_maps': [16, 32, 64, 128, 256],
          'seed': 1
           }
previous_dir = None
# previous_dir = '/export/home/elbrion/hecktor/pet_only/results/model_unet_3d_epochs_150_lr_0.0001_batch_size_2_feat_maps_[16, 32, 64, 128, 256]_seed_1/fold_CHGJ'
train(partition, previous_dir, gpu, dest_dir, fold_name, params)

# src_dir = '/export/home/elbrion/hecktor/pet_only/results/model_unet_3d_epochs_100_lr_0.0001_batch_size_2_feat_maps_[16, 32, 64, 128, 256]_seed_1/fold_CHGJ'
# evaluate(partition['val'], src_dir, gpu, save_predictions=False)