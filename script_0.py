from models import *
from data_generators import *
from training import *
# from evaluation import *

data_dir = '/DATA/public/hecktor_data'

patients_old = np.load(data_dir + '/patients.npy')
patients = {}
for center in ['CHGJ', 'CHMR', 'CHUM', 'CHUS']:
    patients[center] = [elem for elem in patients_old if elem.startswith(center)]

# test_center = 'CHMR'
test_center = 'CHGJ'
partition = {'train': patients['CHMR']+patients['CHUM']+patients['CHUS'],
             'val': patients[test_center]}
print('partition', partition)
print('n_train', len(partition['train']))
print('n_val', len(partition['val']))
gpu = 0
fold_name = test_center

dest_dir = 'results'

for seed in np.arange(0,100):
    params = {'model': 'unet_3d',
              'modality': 'ct',
              'loss': 'dice_loss',
              'epochs': 10,
              'lr': 1e-4,
              'batch_size': 2,
              'feat_maps': [16, 32, 64, 128, 256],
              'seed': seed
               }
    previous_dir = None
    # previous_dir = '/export/home/elbrion/hecktor/dual_modality/results/model_unet_3d_modality_dual_epochs_10_lr_0.0001_batch_size_2_feat_maps_[16, 32, 64, 128, 256]_seed_8/fold_CHMR'
    train(partition, previous_dir, gpu, dest_dir, fold_name, params)


    previous_dir = 'results/model_unet_3d_modality_ct_loss_dice_loss_epochs_10_lr_0.0001_batch_size_2_feat_maps_[16, 32, 64, 128, 256]_seed_'+str(seed)+'/fold_'+test_center
    history = pickle.load(open(previous_dir + '/history.p', "rb"))

    print('loss', history['loss'][-1])
    if history['loss'][-1]<-0.1:
        params['epochs'] = 100
        train(partition, previous_dir, gpu, dest_dir, fold_name, params)