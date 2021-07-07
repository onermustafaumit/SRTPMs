import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='')

parser.add_argument('--imaging_file_primary_solid_tumor', default='cropped_patches_info_patient_id_primary_solid_tumor.txt', help='Cropped patches info file for tumor samples', dest='imaging_file_primary_solid_tumor')
parser.add_argument('--imaging_file_solid_tissue_normal', default='cropped_patches_info_patient_id_solid_tissue_normal.txt', help='Cropped patches info file for normal samples', dest='imaging_file_solid_tissue_normal')
parser.add_argument('--genomic_file', default='purity_ABSOLUTE.txt', help='Genomic tumor purity values', dest='genomic_file')
parser.add_argument('--dataset_dir', default='../dataset/all_patches__level1__stride512__size512', help='Dataset directory', dest='dataset_dir')

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.dataset_dir):
	try:
		os.makedirs(FLAGS.dataset_dir)
	except:
		print("An exception occurred!")


# get purity values
genomic_data = np.loadtxt(FLAGS.genomic_file, comments='#', delimiter='\t',dtype=str)
patient_ids_genomic = genomic_data[:,0]
purity_scores = np.asarray(genomic_data[:,1],dtype=np.float32)
purity_scores = np.around(purity_scores, decimals=3)

group_ids = np.asarray(purity_scores//0.1,dtype=int)
genomic_set = set(list(patient_ids_genomic))


##### primary_solid_tumor #####
print('##### primary_solid_tumor #####')

dataset_file = '{}/all_patients_info_file.txt'.format(FLAGS.dataset_dir)
with open(dataset_file, 'w') as f_dataset_file:
	f_dataset_file.write('# patient_id\tnum_patches\tpurity_score\tgroup_id\n')
	
imaging_data = np.loadtxt(FLAGS.imaging_file_primary_solid_tumor, comments='#', delimiter='\t',dtype=str)
patient_ids_imaging = imaging_data[:,0]
num_patches = np.asarray(imaging_data[:,1:],dtype=int)
num_target_patches = num_patches[:,0]

print('patient_ids_imaging shape:{}'.format(patient_ids_imaging.shape))
print('patient_ids_genomic shape:{}'.format(patient_ids_genomic.shape))

imaging_set = set(list(patient_ids_imaging))

print('len(imaging_set - genomic_set):{}'.format(len(imaging_set - genomic_set)))
print('imaging_set - genomic_set:{}'.format(imaging_set - genomic_set))
print('len(genomic_set - imaging_set):{}'.format(len(genomic_set - imaging_set)))
print('genomic_set - imaging_set:{}'.format(genomic_set - imaging_set))
print('len(genomic_set & imaging_set):{}'.format(len(genomic_set & imaging_set)))
print('genomic_set & imaging_set:{}'.format(genomic_set & imaging_set))

for i in range(len(patient_ids_imaging)):
	temp_patient_id = patient_ids_imaging[i]
	temp_num_target_patches = str(num_target_patches[i])
	for j in range(len(patient_ids_genomic)):
		if temp_patient_id != patient_ids_genomic[j]:
			continue

		temp_purity_score = str(purity_scores[j])
		temp_group_id = str(group_ids[j])

		with open(dataset_file, 'a') as f_dataset_file:
			f_dataset_file.write(temp_patient_id + '\t' + temp_num_target_patches + '\t' + temp_purity_score + '\t' + temp_group_id + '\n')



##### solid_tissue_normal #####
print('##### solid_tissue_normal #####')

dataset_file = '{}/all_patients_solid_tissue_normal_info_file.txt'.format(FLAGS.dataset_dir)
with open(dataset_file, 'w') as f_dataset_file:
	f_dataset_file.write('# patient_ids\tnum_patches\tpurity_scores\tgroup_id\n')
	
imaging_data = np.loadtxt(FLAGS.imaging_file_solid_tissue_normal, comments='#', delimiter='\t',dtype=str)
patient_ids_imaging = imaging_data[:,0]
num_patches = np.asarray(imaging_data[:,1:],dtype=int)
num_target_patches = num_patches[:,0]

print('patient_ids_imaging shape:{}'.format(patient_ids_imaging.shape))
print('patient_ids_genomic shape:{}'.format(patient_ids_genomic.shape))

imaging_set = set(list(patient_ids_imaging))

print('len(imaging_set - genomic_set):{}'.format(len(imaging_set - genomic_set)))
print('imaging_set - genomic_set:{}'.format(imaging_set - genomic_set))
print('len(genomic_set - imaging_set):{}'.format(len(genomic_set - imaging_set)))
print('genomic_set - imaging_set:{}'.format(genomic_set - imaging_set))
print('len(genomic_set & imaging_set):{}'.format(len(genomic_set & imaging_set)))
print('genomic_set & imaging_set:{}'.format(genomic_set & imaging_set))

for i in range(len(patient_ids_imaging)):
	temp_patient_id = patient_ids_imaging[i]
	temp_num_target_patches = str(num_target_patches[i])
	for j in range(len(patient_ids_genomic)):
		if temp_patient_id != patient_ids_genomic[j]:
			continue

		temp_purity_score = str(0.0)
		temp_group_id = str(-1)

		with open(dataset_file, 'a') as f_dataset_file:
			f_dataset_file.write(temp_patient_id + '\t' + temp_num_target_patches + '\t' + temp_purity_score + '\t' + temp_group_id + '\n')


