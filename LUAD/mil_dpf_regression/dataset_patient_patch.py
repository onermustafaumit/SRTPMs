import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
	def __init__(self, image_dir=None, normal_image_dir=None, dataset_dir=None, dataset_type='test', fold_list=None, patch_size=299):
		self._image_dir = image_dir
		self._normal_image_dir = normal_image_dir
		self._dataset_dir = dataset_dir
		self._dataset_type = dataset_type
		self._fold_list = fold_list
		self._patch_size = patch_size

		self._patient_ids_arr, self._num_patches_arr, self._labels_arr, self._img_dirs_arr = self.read_patient_list()

		self._num_patients = self._patient_ids_arr.shape[0]

		self._current_patient_ind = -1

		self._img_transforms = self.image_transforms()


	@property
	def num_patients(self):
		return self._num_patients

	@property
	def patient_ids_arr(self):
		return self._patient_ids_arr

	@property
	def num_images(self):
		return self._current_num_images

	def __len__(self):
		return self._current_num_images

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

		img_dirs_list = list()
		for i in range(len(patient_ids_list)):
			patient_id = patient_ids_list[i]
			group_id = group_ids_list[i]

			if group_id == -1: 
				img_dirs_list.append('{}/{}'.format(self._normal_image_dir, patient_id))
				patient_ids_list[i] = '{}-11'.format(patient_id)
			else:
				img_dirs_list.append('{}/{}'.format(self._image_dir, patient_id))
				patient_ids_list[i] = '{}-01'.format(patient_id)


		patient_ids_arr = np.array(patient_ids_list)
		# print('patient_ids_arr shape:{}'.format(patient_ids_arr.shape))
		num_patches_arr = np.array(num_patches_list)
		# print('num_patches_arr shape:{}'.format(num_patches_arr.shape))
		labels_arr = np.array(labels_list)
		# print('labels_arr shape:{}'.format(labels_arr.shape))
		group_ids_arr = np.array(group_ids_list)
		# print('group_ids_arr shape:{}'.format(group_ids_arr.shape))
		img_dirs_arr = np.array(img_dirs_list)
		# print('img_dir_arr shape:{}'.format(img_dir_arr.shape))

		return patient_ids_arr, num_patches_arr, labels_arr, img_dirs_arr


	def next_patient(self):

		self._current_patient_ind += 1

		if self._current_patient_ind >= self._num_patients:
			print('Error: current patient index exceeded range!!!')
			sys.exit()

		self._current_num_images = self._num_patches_arr[self._current_patient_ind]


	def image_transforms(self):
		
		img_transforms = transforms.Compose([	
												transforms.CenterCrop(self._patch_size),
												transforms.ToTensor(),
												self.MyNormalizationTransform(),
												])

		return img_transforms


	class MyNormalizationTransform(object):
		def __call__(self, input_tensor):
			mean_tensor = torch.mean(input_tensor).view((1,))#, dim=(0,1,2))
			std_tensor = torch.std(input_tensor).view((1,))#, dim=(0,1,2))

			if 0 in std_tensor:
				std_tensor[0] = 1.0

			return TF.normalize(input_tensor, mean_tensor, std_tensor)


	def get_sample_data(self, img_dir, patch_id):
		img_path = '{}/{}.jpeg'.format(img_dir, patch_id)

		img = Image.open(img_path).convert("RGB")

		img_tensor = self._img_transforms(img)

		return img_tensor


	def __getitem__(self, idx):

		temp_patient_index = self._current_patient_ind

		temp_patient_id = self._patient_ids_arr[temp_patient_index]
		temp_num_patches = self._num_patches_arr[temp_patient_index]
		temp_label = self._labels_arr[temp_patient_index]
		temp_img_dir = self._img_dirs_arr[temp_patient_index]

		temp_sample = self.get_sample_data(img_dir = temp_img_dir, patch_id = idx)

		temp_label = torch.as_tensor([temp_label], dtype=torch.float32)

		return temp_sample, temp_label


def custom_collate_fn(batch):
	sample_tensors_list, label_tensors_list = zip(*batch)

	return torch.stack(sample_tensors_list,dim=0), torch.stack(label_tensors_list,dim=0)


def worker_init_fn(id):
	np.random.seed(torch.initial_seed()&0xffffffff)

