import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF

from kde_np import KDE

class Dataset(torch.utils.data.Dataset):
	def __init__(self, feature_dir=None, image_dir=None, normal_image_dir=None, dataset_dir=None, dataset_type='test', fold_list=None, num_instances=10, num_bins=21, sigma=0.05):
		self._feature_dir = feature_dir
		self._image_dir = image_dir
		self._normal_image_dir = normal_image_dir
		self._dataset_dir = dataset_dir
		self._dataset_type = dataset_type
		self._fold_list = fold_list
		self._num_instances = num_instances
		self._num_bins = num_bins
		self._sigma = sigma

		# KDE calculator
		self._kde_calculator = KDE(num_bins=num_bins, sigma=sigma)

		self._slide_ids_arr, self._start_ind_arr, self._num_patches_arr, self._labels_arr, self._img_dirs_arr, self._feature_dirs_arr = self.read_patient_list()

		self._num_slides = self._slide_ids_arr.shape[0]

		self._current_slide_ind = -1
		self._current_num_features = 0
		self._current_slide_features = None
		self._current_slide_coordinates = None
		self._current_slide_features_dist = None


	@property
	def num_slides(self):
		return self._num_slides

	@property
	def slide_ids_arr(self):
		return self._slide_ids_arr

	@property
	def num_features(self):
		return self._current_num_features

	def __len__(self):
		return self._current_num_features

	def read_patient_list(self):
		patient_ids_list = list()
		num_patches_list = list()
		labels_list = list()
		group_ids_list = list()

		for fold_id in self._fold_list:
			patient_list_filename = '{}/fold{}_info_file.txt'.format(self._dataset_dir,fold_id)

			temp_fold_data = np.loadtxt(patient_list_filename, comments='#', delimiter='\t', dtype=str) # [:2]
			
			temp_patient_ids = np.asarray(temp_fold_data[:,0],dtype=str)
			temp_num_patches = np.asarray(temp_fold_data[:,1],dtype=int)
			temp_labels = np.asarray(temp_fold_data[:,2],dtype=np.float32)
			temp_group_ids = np.asarray(temp_fold_data[:,3],dtype=int)

			patient_ids_list += list(temp_patient_ids)
			num_patches_list += list(temp_num_patches)
			labels_list += list(temp_labels)
			group_ids_list += list(temp_group_ids)


		##### include normal data #####
		before_count = len(patient_ids_list)

		patient_list_filename = '{}/all_patients_solid_tissue_normal_info_file.txt'.format(self._dataset_dir)

		temp_fold_data = np.loadtxt(patient_list_filename, comments='#', delimiter='\t', dtype=str)
		
		temp_patient_ids = np.asarray(temp_fold_data[:,0],dtype=str)
		temp_num_patches = np.asarray(temp_fold_data[:,1],dtype=int)
		temp_labels = np.asarray(temp_fold_data[:,2],dtype=np.float32)
		temp_group_ids = np.asarray(temp_fold_data[:,3],dtype=int)

		patient_ids_list_copy = patient_ids_list.copy()
		for i in range(len(patient_ids_list_copy)):
			patient_id = patient_ids_list_copy[i]

			ind = np.where(temp_patient_ids == patient_id)[0]
			if ind.size > 0:
				patient_ids_list.append(temp_patient_ids[ind[0]])
				num_patches_list.append(temp_num_patches[ind[0]])
				labels_list.append(temp_labels[ind[0]])
				group_ids_list.append(temp_group_ids[ind[0]])

		after_count = len(patient_ids_list)
		print('num_normals: {}'.format(after_count - before_count))
		##### include normal data #####

		slide_ids_list = []
		start_ind_list = []
		num_patches_slide_list = []
		labels_slide_list = []
		img_dirs_list = []
		feature_dirs_list = []
		for i in range(len(patient_ids_list)):
			patient_id = patient_ids_list[i]
			label = labels_list[i]
			group_id = group_ids_list[i]

			if group_id == -1:			
				# get slide_ids and corresponding patch info for normal samples
				patch_info_file = '{}/{}/cropped_patches_filelist.txt'.format(self._normal_image_dir,patient_id)
				patch_info = np.loadtxt(patch_info_file, skiprows=1, comments='#', delimiter='\t', dtype=str)

				temp_wsi_id_arr, temp_start_ind_arr, temp_num_patches_arr = np.unique(patch_info[:,1], return_index=True, return_counts=True)
				
				for w in range(temp_wsi_id_arr.shape[0]):
					temp_wsi_id = temp_wsi_id_arr[w]
					temp_start_ind = temp_start_ind_arr[w]
					temp_num_patches = temp_num_patches_arr[w]

					patient_ids_list[i] = '{}-11'.format(patient_id)
					slide_ids_list.append(temp_wsi_id)
					start_ind_list.append(temp_start_ind)
					num_patches_slide_list.append(temp_num_patches)
					labels_slide_list.append(label)
					img_dirs_list.append('{}/{}'.format(self._normal_image_dir, patient_id))
					feature_dirs_list.append('{}/{}'.format(self._feature_dir, patient_ids_list[i]))

			else:
				# get slide_ids and corresponding patch info for normal samples
				patch_info_file = '{}/{}/cropped_patches_filelist.txt'.format(self._image_dir,patient_id)
				patch_info = np.loadtxt(patch_info_file, skiprows=1, comments='#', delimiter='\t', dtype=str)

				temp_wsi_id_arr, temp_start_ind_arr, temp_num_patches_arr = np.unique(patch_info[:,1], return_index=True, return_counts=True)
				
				for w in range(temp_wsi_id_arr.shape[0]):
					temp_wsi_id = temp_wsi_id_arr[w]
					temp_start_ind = temp_start_ind_arr[w]
					temp_num_patches = temp_num_patches_arr[w]

					patient_ids_list[i] = '{}-01'.format(patient_id)
					slide_ids_list.append(temp_wsi_id)
					start_ind_list.append(temp_start_ind)
					num_patches_slide_list.append(temp_num_patches)
					labels_slide_list.append(label)
					img_dirs_list.append('{}/{}'.format(self._image_dir, patient_id))
					feature_dirs_list.append('{}/{}'.format(self._feature_dir, patient_ids_list[i]))
				

		patient_ids_arr = np.array(patient_ids_list)
		# print('patient_ids_arr shape:{}'.format(patient_ids_arr.shape))
		num_patches_arr = np.array(num_patches_list)
		# print('num_patches_arr shape:{}'.format(num_patches_arr.shape))
		labels_arr = np.array(labels_list)
		# print('labels_arr shape:{}'.format(labels_arr.shape))
		group_ids_arr = np.array(group_ids_list)
		# print('group_ids_arr shape:{}'.format(group_ids_arr.shape))
		
		
		slide_ids_arr = np.array(slide_ids_list)
		print('slide_ids_arr shape:{}'.format(slide_ids_arr.shape))
		start_ind_arr = np.array(start_ind_list)
		print('start_ind_arr shape:{}'.format(start_ind_arr.shape))
		num_patches_slide_arr = np.array(num_patches_slide_list)
		print('num_patches_slide_arr shape:{}'.format(num_patches_slide_arr.shape))
		labels_slide_arr = np.array(labels_slide_list)
		print('labels_slide_arr shape:{}'.format(labels_slide_arr.shape))
		img_dirs_arr = np.array(img_dirs_list)
		print('img_dirs_arr shape:{}'.format(img_dirs_arr.shape))
		feature_dirs_arr = np.array(feature_dirs_list)
		print('feature_dirs_arr shape:{}'.format(feature_dirs_arr.shape))

		return slide_ids_arr, start_ind_arr, num_patches_slide_arr, labels_slide_arr, img_dirs_arr, feature_dirs_arr


	def read_feature_data(self):
		temp_index = self._current_slide_ind
		temp_slide_id = self._slide_ids_arr[temp_index]
		temp_start_ind = self._start_ind_arr[temp_index]
		temp_num_patches = self._num_patches_arr[temp_index]
		temp_label = self._labels_arr[temp_index]
		temp_img_dir = self._img_dirs_arr[temp_index]
		temp_feature_dir = self._feature_dirs_arr[temp_index]

		patch_indices = np.arange(temp_start_ind,temp_start_ind+temp_num_patches)

		temp_patient_id = temp_slide_id[:15]
		

		data_file = '{}/extracted_features_{}.txt'.format(temp_feature_dir,temp_patient_id)
		data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=np.float32)

		return data[patch_indices]


	def read_coordinates_data(self):
		temp_index = self._current_slide_ind
		temp_slide_id = self._slide_ids_arr[temp_index]
		temp_start_ind = self._start_ind_arr[temp_index]
		temp_num_patches = self._num_patches_arr[temp_index]
		temp_label = self._labels_arr[temp_index]
		temp_img_dir = self._img_dirs_arr[temp_index]
		temp_feature_dir = self._feature_dirs_arr[temp_index]

		patch_indices = np.arange(temp_start_ind,temp_start_ind+temp_num_patches)
		
		patch_info_file = '{}/cropped_patches_filelist.txt'.format(temp_img_dir)
		patch_info = np.loadtxt(patch_info_file, skiprows=1, comments='#', delimiter='\t', dtype=str)

		coordinates_data = np.asarray(patch_info[:,2:4],dtype=int)

		return coordinates_data[patch_indices]


	def next_slide(self):

		self._current_slide_ind += 1
		# print(self._slide_ids_arr[self._current_slide_ind])

		if self._current_slide_ind >= self._num_slides:
			print('Error: current patient index exceeded range!!!')
			sys.exit()

		self._current_num_features = self._num_patches_arr[self._current_slide_ind]
		# print(self._current_num_features)

		self._current_slide_features = self.read_feature_data()

		self._current_slide_coordinates = self.read_coordinates_data()

		if self._current_num_features < self._num_instances:
			self._current_slide_features_dist = self._kde_calculator.calculate(self._current_slide_features[np.newaxis,:,:])
		else:
			self._current_slide_features_dist = None



	def get_sample_data(self, idx):

		current_patch_coordinates = self._current_slide_coordinates[idx]
		# print(current_patch_coordinates)

		if self._num_instances == 1:
			current_patch_features = self._current_slide_features[idx]
			# print(current_patch_features.shape)
			temp_distributions = self._kde_calculator.calculate(current_patch_features[np.newaxis,np.newaxis,:])

		elif self._current_num_features < self._num_instances:
			# get distribution with all patches
			temp_distributions = self._current_slide_features_dist

		else:
			# find self._num_instances closest patches
			distances = np.sum((self._current_slide_coordinates-current_patch_coordinates)**2,axis=1)			
			closest_patch_indices = np.argsort(distances)[:self._num_instances]
			# print(self._current_slide_coordinates[closest_patch_indices])

			temp_features = self._current_slide_features[closest_patch_indices]

			# get distribution
			temp_distributions = self._kde_calculator.calculate(temp_features[np.newaxis,:,:])

		return torch.from_numpy(temp_distributions), torch.from_numpy(current_patch_coordinates)
		

	def __getitem__(self, idx):
		temp_index = self._current_slide_ind
		
		temp_label = self._labels_arr[temp_index]
		temp_label = torch.as_tensor([temp_label], dtype=torch.float32)

		temp_sample, current_patch_coordinates = self.get_sample_data(idx = idx)

		return temp_sample, temp_label, current_patch_coordinates


def custom_collate_fn(batch):
	sample_tensors_list, label_tensors_list, coordinates_tensors_list = zip(*batch)

	return torch.stack(sample_tensors_list,dim=0), torch.stack(label_tensors_list,dim=0), torch.stack(coordinates_tensors_list,dim=0)


def worker_init_fn(id):
	np.random.seed(torch.initial_seed()&0xffffffff)

