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

slide_ids_list = [d for d in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, d))]

slide_ids_arr = np.asarray(sorted(slide_ids_list))
num_slides = slide_ids_arr.shape[0]
print("Data - num_slides: {}".format(num_slides))

# initiate bag predictions file
bag_predictions_file = '{}/bag_predictions_all_slides.txt'.format(data_folder_path)
with open(bag_predictions_file,'w') as f_bag_predictions_file:
	f_bag_predictions_file.write('# bag_id\t')
	f_bag_predictions_file.write('truth\t')
	f_bag_predictions_file.write('pred')
	f_bag_predictions_file.write('\n')


# initiate slide predictions file - mpp: mean predicted purity
slide_predictions_file_mpp = '{}/slide_predictions_mpp.txt'.format(data_folder_path)
with open(slide_predictions_file_mpp,'w') as f_slide_predictions_file_mpp:
	f_slide_predictions_file_mpp.write('# slide_id\t')
	f_slide_predictions_file_mpp.write('truth\t')
	f_slide_predictions_file_mpp.write('pred')
	f_slide_predictions_file_mpp.write('\n')


bag_ids_list = []
bag_truths_list = []
bag_probs_list = []
slide_truths_list = []
slide_probs_mpp_list = []
for i in range(num_slides):

	slide_id = slide_ids_arr[i]
	print('Slide {}/{}: {}'.format(i+1,num_slides,slide_id))

	slide_data_folder_path = '{}/{}'.format(data_folder_path,slide_id)

	test_metrics_filename = '{}/bag_predictions_{}.txt'.format(slide_data_folder_path,slide_id)

	test_metrics_data = np.loadtxt(test_metrics_filename, delimiter='\t', comments='#', dtype=str)
	bag_ids_data = np.asarray(test_metrics_data[:,0],dtype=str)
	truths_data = np.asarray(test_metrics_data[:,1],dtype=float)
	probs_data = np.asarray(test_metrics_data[:,2],dtype=float)

	bag_ids_list.append(bag_ids_data)
	bag_truths_list.append(truths_data)
	bag_probs_list.append(probs_data)

	slide_truths = truths_data[0]
	slide_probs_mpp = np.mean(probs_data,axis=0)

	slide_truths_list.append(slide_truths)
	slide_probs_mpp_list.append(slide_probs_mpp)

	with open(slide_predictions_file_mpp,'a') as f_slide_predictions_file_mpp:
		f_slide_predictions_file_mpp.write('{}\t'.format(slide_id))
		f_slide_predictions_file_mpp.write('{:.3f}\t'.format(slide_truths))
		f_slide_predictions_file_mpp.write('{:.3f}\n'.format(slide_probs_mpp))


bag_ids_arr = np.concatenate(bag_ids_list, axis=0)
print('bag_ids_arr.shape:{}'.format(bag_ids_arr.shape))
bag_truths_arr = np.concatenate(bag_truths_list, axis=0)
print('bag_truths_arr.shape:{}'.format(bag_truths_arr.shape))
bag_probs_arr = np.concatenate(bag_probs_list, axis=0)
print('bag_probs_arr.shape:{}'.format(bag_probs_arr.shape))
slide_truths_arr = np.stack(slide_truths_list, axis=0)
print('slide_truths_arr.shape:{}'.format(slide_truths_arr.shape))
slide_probs_mpp_arr = np.stack(slide_probs_mpp_list, axis=0)
print('slide_probs_mpp_arr.shape:{}'.format(slide_probs_mpp_arr.shape))

# write bag predictions of all slides into single file
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


# collect slide level statistics
temp_truth = slide_truths_arr
temp_prob = slide_probs_mpp_arr

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
fig_filename = '{}/slide_level_scatter_plot_mpp.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)

# plt.show()

plt.close('all')




