import numpy as np
import argparse
import os
import sys
from os import path

import matplotlib.pyplot as plt

from scipy import stats

plt.rcParams.update({'font.size':8, 'font.family':'Times New Roman'})


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='data folder path', dest='data_folder_path')
FLAGS = parser.parse_args()

data_folder_path = FLAGS.data_folder_path

patient_ids_list = [d for d in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, d))]

patient_ids_arr = np.asarray(sorted(patient_ids_list))
num_patients = patient_ids_arr.shape[0]
print("Data - num_patients: {}".format(num_patients))

# initiate bag predictions file
bag_predictions_file = '{}/bag_predictions_all_patients.txt'.format(data_folder_path)
with open(bag_predictions_file,'w') as f_bag_predictions_file:
	f_bag_predictions_file.write('# bag_id\t')
	f_bag_predictions_file.write('truth\t')
	f_bag_predictions_file.write('pred')
	f_bag_predictions_file.write('\n')


# initiate patient predictions file - mpp: mean predicted purity
patient_predictions_file_mpp = '{}/patient_predictions_mpp.txt'.format(data_folder_path)
with open(patient_predictions_file_mpp,'w') as f_patient_predictions_file_mpp:
	f_patient_predictions_file_mpp.write('# patient_id\t')
	f_patient_predictions_file_mpp.write('truth\t')
	f_patient_predictions_file_mpp.write('pred')
	f_patient_predictions_file_mpp.write('\n')


bag_ids_list = []
bag_truths_list = []
bag_probs_list = []
patient_truths_list = []
patient_probs_mpp_list = []
for i in range(num_patients):

	patient_id = patient_ids_arr[i]
	print('Patient {}/{}: {}'.format(i+1,num_patients,patient_id))

	patient_data_folder_path = '{}/{}'.format(data_folder_path,patient_id)

	test_metrics_filename = '{}/bag_predictions_{}.txt'.format(patient_data_folder_path,patient_id)

	test_metrics_data = np.loadtxt(test_metrics_filename, delimiter='\t', comments='#', dtype=str)
	bag_ids_data = np.asarray(test_metrics_data[:,0],dtype=str)
	truths_data = np.asarray(test_metrics_data[:,1],dtype=float)
	probs_data = np.asarray(test_metrics_data[:,2],dtype=float)

	bag_ids_list.append(bag_ids_data)
	bag_truths_list.append(truths_data)
	bag_probs_list.append(probs_data)

	patient_truths = truths_data[0]
	patient_probs_mpp = np.mean(probs_data,axis=0)

	patient_truths_list.append(patient_truths)
	patient_probs_mpp_list.append(patient_probs_mpp)

	with open(patient_predictions_file_mpp,'a') as f_patient_predictions_file_mpp:
		f_patient_predictions_file_mpp.write('{}\t'.format(patient_id))
		f_patient_predictions_file_mpp.write('{:.3f}\t'.format(patient_truths))
		f_patient_predictions_file_mpp.write('{:.3f}\n'.format(patient_probs_mpp))


bag_ids_arr = np.concatenate(bag_ids_list, axis=0)
print('bag_ids_arr.shape:{}'.format(bag_ids_arr.shape))
bag_truths_arr = np.concatenate(bag_truths_list, axis=0)
print('bag_truths_arr.shape:{}'.format(bag_truths_arr.shape))
bag_probs_arr = np.concatenate(bag_probs_list, axis=0)
print('bag_probs_arr.shape:{}'.format(bag_probs_arr.shape))
patient_truths_arr = np.stack(patient_truths_list, axis=0)
print('patient_truths_arr.shape:{}'.format(patient_truths_arr.shape))
patient_probs_mpp_arr = np.stack(patient_probs_mpp_list, axis=0)
print('patient_probs_mpp_arr.shape:{}'.format(patient_probs_mpp_arr.shape))

# write bag predictions of all patients into single file
num_patches = bag_truths_arr.shape[0]
with open(bag_predictions_file,'a') as f_bag_predictions_file:
	for i in range(num_patches):
		f_bag_predictions_file.write('{}\t'.format(bag_ids_arr[i]))
		f_bag_predictions_file.write('{:.3f}\t'.format(bag_truths_arr[i]))
		f_bag_predictions_file.write('{:.3f}\n'.format(bag_probs_arr[i]))


# collect bag level statistics
temp_truth = bag_truths_arr
temp_prob = bag_probs_arr

rho, pval = stats.spearmanr(temp_prob,temp_truth)	
print('spearman rho:{:.3f} - pval:{:.1e}'.format(rho,pval))

title_text = r'$\rho$ = {:.3f}, p = {:.1e}'.format(rho, pval)

fig, ax = plt.subplots(figsize=(3,3))

ax.plot([0,1], [0,1], linestyle='--', color='red', zorder=1)
ax.scatter(temp_prob,temp_truth, color='k', zorder=2, alpha=0.7, s=16)
ax.set_xlabel('MIL prediction')
ax.set_ylabel('Genomic tumor purity')
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
ax.set_axisbelow(True)
ax.grid()
ax.set_title(title_text)

fig.tight_layout()
fig_filename = '{}/bag_level_scatter_plot.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)


# collect patient level statistics
temp_truth = patient_truths_arr
temp_prob = patient_probs_mpp_arr

rho, pval = stats.spearmanr(temp_prob,temp_truth)	
print('spearman rho:{:.3f} - pval:{:.1e}'.format(rho,pval))

title_text = r'$\rho$ = {:.3f}, p = {:.1e}'.format(rho, pval)

fig, ax = plt.subplots(figsize=(3,3))

ax.plot([0,1], [0,1], linestyle='--', color='red', zorder=1)
ax.scatter(temp_prob,temp_truth, color='k', zorder=2, alpha=0.7, s=16)
ax.set_xlabel('MIL prediction')
ax.set_ylabel('Genomic tumor purity')
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
ax.set_axisbelow(True)
ax.grid()
ax.set_title(title_text)

fig.tight_layout()
fig_filename = '{}/patient_level_scatter_plot_mpp.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)

plt.show()

plt.close('all')




