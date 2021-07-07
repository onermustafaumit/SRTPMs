import numpy as np
import argparse
import os
import sys
import time

from model import Model
from dataset_distribution_closest_patches import Dataset, custom_collate_fn, worker_init_fn

import torch
import torch.utils.data
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='')

parser.add_argument('--init_model_file', default='', help='the path of initial model file', dest='init_model_file')
parser.add_argument('--image_dir', default='../Images/all_cropped_patches_primary_solid_tumor__level1__stride512__size512', help='Image directory for tumor patches', dest='image_dir')
parser.add_argument('--normal_image_dir', default='../Images/all_cropped_patches_solid_tissue_normal__level1__stride512__size512', help='Image directory for normal patches', dest='normal_image_dir')
parser.add_argument('--feature_dir', default='extracted_features', help='directory of extracted feature files', dest='feature_dir')
parser.add_argument('--dataset_dir', default='../dataset/all_patches__level1__stride512__size512', help='dataset info folder', dest='dataset_dir')
parser.add_argument('--dataset_type', default='test', help='Dataset type: test, valid, train', dest='dataset_type')
parser.add_argument('--patch_size', default='299', type=int, help='patch size', dest='patch_size')
parser.add_argument('--num_instances', default='16', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_features', default='128', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_bins', default='21', type=int, help='number of bins in distribution pooling filter', dest='num_bins')
parser.add_argument('--sigma', default='0.05', type=float, help='sigma in distribution pooling filter', dest='sigma')
parser.add_argument('--num_classes', default='1', type=int, help='number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='64', type=int, help='batch size', dest='batch_size')
parser.add_argument('--out_dir', default='patch_scores', help='directory to store patch score files', dest='out_dir')
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

# init_model_file: saved_models/model_weights__2020_12_04__15_36_52__10.pth
model_name = FLAGS.init_model_file.split('__')[-3] + '__' + FLAGS.init_model_file.split('__')[-2] + '__' + FLAGS.init_model_file.split('__')[-1][:-4]
data_folder_path = '{}__{}/{}/{}'.format(FLAGS.out_dir,FLAGS.num_instances,model_name,FLAGS.dataset_type)
if not os.path.exists(data_folder_path):
	os.makedirs(data_folder_path)

feature_dir = '{}/{}/{}'.format(FLAGS.feature_dir,model_name,FLAGS.dataset_type)
dataset = Dataset(feature_dir=feature_dir, image_dir=FLAGS.image_dir, normal_image_dir=FLAGS.normal_image_dir, dataset_dir=FLAGS.dataset_dir, dataset_type=FLAGS.dataset_type, fold_list=fold_list, num_instances=FLAGS.num_instances, num_bins=FLAGS.num_bins, sigma=FLAGS.sigma)

num_slides = dataset.num_slides
slide_ids_arr = dataset.slide_ids_arr

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=10, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)


# print model parameters
print('# Model parameters:')
for key in FLAGS_dict.keys():
	print('# {} = {}'.format(key, FLAGS_dict[key]))

print("# num_slides: {}".format(num_slides))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, num_instances=FLAGS.num_instances, num_features=FLAGS.num_features, num_bins=FLAGS.num_bins, sigma=FLAGS.sigma)
model.to(device)

state_dict = torch.load(FLAGS.init_model_file, map_location=device)
model.load_state_dict(state_dict['model_state_dict'])
print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))

model.eval()
with torch.no_grad():

	for i in range(num_slides):
		dataset.next_slide()

		slide_id = slide_ids_arr[i]
		print('Slide {}/{}: {}'.format(i+1,num_slides,slide_id))

		slide_data_folder_path = '{}/{}'.format(data_folder_path,slide_id)
		if not os.path.exists(slide_data_folder_path):
			os.makedirs(slide_data_folder_path)

		out_file = '{}/patch_scores_{}.txt'.format(slide_data_folder_path,slide_id)
		with open(out_file, 'w') as f_out_file:
			f_out_file.write('# row_id\tcol_id\tscore\n')

		for distributions, targets, coordinates in data_loader:
			distributions = torch.flatten(distributions, 1)
			distributions = distributions.to(device)
			# print(distributions.size())
			# print(targets.size())

			scores_out = model._representation_transformation(distributions)
			scores_out = scores_out.cpu().numpy()
			# print('scores_out.shape: {}'.format(scores_out.shape))
			# print(scores_out)

			coordinates = coordinates.numpy()
			# print(coordinates)

			with open(out_file, 'a') as f_out_file:
				for j in range(scores_out.shape[0]):
					f_out_file.write('{}\t{}\t{:.4f}\n'.format(coordinates[j][0],coordinates[j][1],scores_out[j][0]))

print('Finished!!!')

