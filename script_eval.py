from evaluation import *

data_dir = '/DATA/public/hecktor_data'

patients_old = np.load(data_dir + '/patients.npy')
patients = {}
for center in ['CHGJ', 'CHMR', 'CHUM', 'CHUS']:
    patients[center] = [elem for elem in patients_old if elem.startswith(center)]

filenames = patients['CHUS']

src_dir = '/export/home/elbrion/hecktor/dual_modality/results/model_unet_3d_modality_dual_epochs_100_lr_0.0001_batch_size_2_feat_maps_[16, 32, 64, 128, 256]_seed_1/fold_CHUS'
gpu = 0
modality = 'dual'
evaluate(filenames, src_dir, gpu, modality, save_predictions=False)