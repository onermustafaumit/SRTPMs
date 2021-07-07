import numpy as np
import argparse
from datetime import datetime
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', default='../dataset/all_patches__level1__stride512__size512', help='', dest='dataset_dir')
FLAGS = parser.parse_args()

labels_file = '{}/all_patients_info_file.txt'.format(FLAGS.dataset_dir)

with open(labels_file, 'r') as f_labels_file:
	header = next(f_labels_file)

print(header)

data = np.loadtxt(labels_file, comments='#', delimiter='\t', dtype=str)
print(data.shape)

num_patches = np.asarray(data[:,1],dtype=int)
group_ids = np.asarray(data[:,3],dtype=int)
num_samples = group_ids.shape[0]

unique_ids = np.unique(group_ids)
num_unique_ids = len(unique_ids)
print(unique_ids)

members = np.zeros((num_samples,num_unique_ids),dtype=int)
for i in range(num_unique_ids):
	temp_unique_id = unique_ids[i]

	for j in range(num_samples):
		if group_ids[j] == temp_unique_id:
			members[j,i] = 1

fold_ind_dict={0:[],1:[],2:[],3:[],4:[]}
remainders_list = []
for temp_unique_id in unique_ids:
	temp_indices = np.where(group_ids==temp_unique_id)[0]
	np.random.shuffle(temp_indices)

	temp_num_samples = temp_indices.shape[0]
	temp_num_samples_per_fold = temp_num_samples // 5
	remainders_list += list(temp_indices[temp_num_samples_per_fold*5:])

	# update fold_ind_dict
	for i in range(5):
		fold_ind_dict[i] += list(temp_indices[i*temp_num_samples_per_fold:(i+1)*temp_num_samples_per_fold])

print(fold_ind_dict)
print(remainders_list)

remainders_arr = np.array(remainders_list)

for i,temp_ind in enumerate(remainders_arr):
	fold_ind_dict[i%5].append(temp_ind)

print(fold_ind_dict)

temp_list = ['fold#','num_patches'] + list(unique_ids)
print(temp_list)

for i in range(5):
	temp_indices = fold_ind_dict[i]

	temp_fold_data = data[temp_indices,:]

	temp_num_patches = np.asarray(temp_fold_data[:,1],dtype=int)
	temp_total_num_patches = np.sum(temp_num_patches)

	temp_members = members[temp_indices,:]
	temp_num_members_group_based = np.sum(temp_members,axis=0)

	temp_list = ['fold{}'.format(i),'{}'.format(temp_total_num_patches)] + list(temp_num_members_group_based)

	print(temp_list)

	filename = '{}/fold{}_info_file.txt'.format(FLAGS.dataset_dir,i)

	with open(filename,'w') as f_filename:
		f_filename.write(header)

	with open(filename,'a') as f_filename:
		np.savetxt(f_filename, temp_fold_data, fmt='%s', delimiter='\t')



