import numpy as np
import argparse
import os
import sys
import time

from model import Model
from dataset_patient_patch import Dataset, custom_collate_fn, worker_init_fn

import torch
import torch.utils.data
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='')

parser.add_argument('--init_model_file', default='', help='the path of initial model file', dest='init_model_file')
parser.add_argument('--image_dir', default='../Images/all_cropped_patches_primary_solid_tumor__level1__stride512__size512', help='Image directory for tumor patches', dest='image_dir')
parser.add_argument('--normal_image_dir', default='../Images/all_cropped_patches_solid_tissue_normal__level1__stride512__size512', help='Image directory for normal patches', dest='normal_image_dir')
parser.add_argument('--dataset_dir', default='../dataset/all_patches__level1__stride512__size512', help='dataset info folder', dest='dataset_dir')
parser.add_argument('--dataset_type', default='test', help='Dataset type: test, valid, train', dest='dataset_type')
parser.add_argument('--patch_size', default='299', type=int, help='patch size', dest='patch_size')
parser.add_argument('--num_instances', default='200', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_features', default='128', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_bins', default='21', type=int, help='number of bins in distribution pooling filter', dest='num_bins')
parser.add_argument('--sigma', default='0.05', type=float, help='sigma in distribution pooling filter', dest='sigma')
parser.add_argument('--num_classes', default='1', type=int, help='number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='64', type=int, help='batch size', dest='batch_size')
parser.add_argument('--out_dir', default='extracted_features', help='directory to store extracted feature files', dest='out_dir')
parser.add_argument('--valid_fold', default=3, type=int, help='id of fold to be used as validation set', dest='valid_fold')
parser.add_argument('--test_fold', default=4, type=int, help='id of fold to be used as test set', dest='test_fold')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

train_fold_list = np.arange(5)
train_fold_list = np.delete(train_fold_list, [FLAGS.valid_fold,FLAGS.test_fold])
valid_fold_list = [FLAGS.valid_fold]
test_fold_list = [FLAGS.test_fold]

if FLAGS.dataset_type == 'train':
	fold_list = train_fold_list
elif FLAGS.dataset_type == 'valid':
	fold_list = valid_fold_list
elif FLAGS.dataset_type == 'test':
	fold_list = test_fold_list

dataset = Dataset(image_dir=FLAGS.image_dir, normal_image_dir=FLAGS.normal_image_dir, dataset_dir=FLAGS.dataset_dir, dataset_type=FLAGS.dataset_type, patch_size=FLAGS.patch_size, fold_list=fold_list)
num_patients = dataset.num_patients
patient_ids_arr = dataset.patient_ids_arr
print("Data - num_patients: {}".format(num_patients))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

# init_model_file: saved_models/model_weights__2020_12_04__15_36_52__10.pth
model_name = FLAGS.init_model_file.split('__')[-3] + '__' + FLAGS.init_model_file.split('__')[-2] + '__' + FLAGS.init_model_file.split('__')[-1][:-4]
data_folder_path = '{}/{}/{}'.format(FLAGS.out_dir,model_name,FLAGS.dataset_type)
if not os.path.exists(data_folder_path):
	os.makedirs(data_folder_path)


# print model parameters
print('# Model parameters:')
for key in FLAGS_dict.keys():
	print('# {} = {}'.format(key, FLAGS_dict[key]))

print("# num_patients: {}".format(num_patients))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, num_instances=FLAGS.num_instances, num_features=FLAGS.num_features, num_bins=FLAGS.num_bins, sigma=FLAGS.sigma)
model.to(device)

state_dict = torch.load(FLAGS.init_model_file, map_location=device)
model.load_state_dict(state_dict['model_state_dict'])
print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))

model.eval()
with torch.no_grad():

	for i in range(num_patients):
		dataset.next_patient()

		patient_id = patient_ids_arr[i]
		print('Patient {}/{}: {}'.format(i+1,num_patients,patient_id))

		patient_data_folder_path = '{}/{}'.format(data_folder_path,patient_id)
		if not os.path.exists(patient_data_folder_path):
			os.makedirs(patient_data_folder_path)

		out_file = '{}/extracted_features_{}.txt'.format(patient_data_folder_path,patient_id)

		for images, targets in data_loader:
			images = images.to(device)
			# print(images.size())
			# print(targets.size())

			features_out = model._feature_extractor(images)
			features_out = features_out.cpu().numpy()
			# print('features_out.shape: {}'.format(features_out.shape))

			with open(out_file, 'ab') as f_out_file:
				np.savetxt(f_out_file, features_out.reshape((-1,FLAGS.num_features)), fmt='%5.4f', delimiter='\t')

print('Feature extraction finished!!!')



