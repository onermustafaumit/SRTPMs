import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, ttest_rel, normaltest, wilcoxon, mannwhitneyu
import math
import sys
import os

plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='')
FLAGS = parser.parse_args()

data_folder_path_slides = FLAGS.data_folder_path
data_folder_path_samples = 'test_metrics/{}'.format('/'.join(data_folder_path_slides.split('/')[1:]))
data_folder_path_slides_processed = FLAGS.data_folder_path + '__processed'
if not os.path.exists(data_folder_path_slides_processed):
	os.makedirs(data_folder_path_slides_processed)

# get samples predictions
sample_pred_dict = dict()

data_file = '{}/patient_predictions_mpp.txt'.format(data_folder_path_samples)
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)
num_samples = data.shape[0]

for i in range(num_samples):
	temp_data = data[i]
	temp_sample_id = temp_data[0]
	temp_truth = float(temp_data[1])
	temp_pred = float(temp_data[2])

	sample_pred_dict[temp_sample_id] = {
											'sample_truth':temp_truth,
											'sample_pred':temp_pred
	}


# get slide predictions
data_file = '{}/slide_predictions_mpp.txt'.format(data_folder_path_slides)
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)
num_slides = data.shape[0]


out_file2 = '{}/statistical_test__compare_top_bottom_slides_of_a_sample__pvalues.txt'.format(data_folder_path_slides_processed)
with open(out_file2, 'w') as f_out_file2:
	f_out_file2.write('# sample_id\tp_val\n')


wilcoxon_p_val_list = []
for i in range(num_slides):
	temp_data = data[i]
	temp_slide_id = temp_data[0]
	temp_sample_id = temp_slide_id[:15]
	temp_truth = float(temp_data[1])
	temp_pred = float(temp_data[2])

	# check temp_sample_id in sample_pred_dict
	if temp_sample_id not in sample_pred_dict:
		print("ERROR: temp_sample_id is not in sample_pred_dict!!!")
		sys.exit()

	# get sample_truth and sample_pred
	sample_truth = sample_pred_dict[temp_sample_id]['sample_truth']
	sample_pred = sample_pred_dict[temp_sample_id]['sample_pred']

	# check temp_truth == sample_truth
	if temp_truth != sample_truth:
		print("ERROR: temp_truth == sample_truth !!!")
		sys.exit()


	# get bag predictions for the slide
	slide_data_folder_path = '{}/{}'.format(data_folder_path_slides,temp_slide_id)
	test_metrics_filename = '{}/bag_predictions_{}.txt'.format(slide_data_folder_path,temp_slide_id)

	test_metrics_data = np.loadtxt(test_metrics_filename, delimiter='\t', comments='#', dtype=str)
	bag_ids_data = np.asarray(test_metrics_data[:,0],dtype=str)
	truths_data = np.asarray(test_metrics_data[:,1],dtype=float)
	probs_data = np.asarray(test_metrics_data[:,2],dtype=float)


	# first slide
	if i%2 == 0:
		slide_id0 = temp_slide_id
		slide_pred_arr0 = probs_data

	# second slide
	else:
		slide_id1 = temp_slide_id
		slide_pred_arr1 = probs_data

		# we got the second slide, so ready to process
		print('Sample {}/{}: {}'.format(i,num_slides-1,temp_sample_id))

		data_folder_path_slides_processed_sample = '{}/{}'.format(data_folder_path_slides_processed,temp_sample_id)
		if not os.path.exists(data_folder_path_slides_processed_sample):
			os.makedirs(data_folder_path_slides_processed_sample)

		# plot hist of slide_preds
		fig0, ax0 = plt.subplots(figsize=(3,3))
		n, bins, patches = ax0.hist(slide_pred_arr0, 20, density=False, facecolor='black', alpha=0.75, label=slide_id0)
		n, bins, patches = ax0.hist(slide_pred_arr1, 20, density=False, facecolor='gray', alpha=0.75, label=slide_id1)
		ax0.set_xlabel('MIL prediction')
		ax0.set_ylabel('# samples')
		# ax0.set_title('',fontsize=10)
		ax0.set_axisbelow(True)
		ax0.grid()

		ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 1.), fancybox=True, shadow=False, ncol=1,fontsize=10,frameon=True)

		fig0.tight_layout()
		fig_filename0 = '{}/slides_predictions__histogram__{}.pdf'.format(data_folder_path_slides_processed_sample,temp_sample_id)
		fig0.savefig(fig_filename0, dpi=200)


		##### statistics on slide predictions #####
		print('##### statistics on slide predictions #####')
		out_file = '{}/statistical_test__compare_top_bottom_slides_of_a_sample__summary__{}.txt'.format(data_folder_path_slides_processed_sample,temp_sample_id)
		with open(out_file, 'w') as f_out_file:
			f_out_file.write('##### statistics on slide predictions #####\n')


		### statistics on slide0 predictions ###
		print('### statistics on slide0 predictions ###')
		mean_slide_pred_arr0 = np.mean(slide_pred_arr0)
		std_slide_pred_arr0 = np.std(slide_pred_arr0)
		median_slide_pred_arr0, Q1_slide_pred_arr0, Q3_slide_pred_arr0 = np.percentile(slide_pred_arr0, (50,25,75), interpolation='nearest')
		print('mean_slide_pred_arr0: {:.3f}, std_slide_pred_arr0: {:.3f}, median_slide_pred_arr0: {:.3f}, Q1_slide_pred_arr0: {:.3f}, Q3_slide_pred_arr0: {:.3f}'.format(mean_slide_pred_arr0,std_slide_pred_arr0,median_slide_pred_arr0,Q1_slide_pred_arr0,Q3_slide_pred_arr0))
		with open(out_file, 'a') as f_out_file:
			f_out_file.write('### statistics on slide0 predictions ###\n')
			f_out_file.write('# mean\tstd\tmedian\tQ1\tQ3\n')
			f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(mean_slide_pred_arr0,std_slide_pred_arr0,median_slide_pred_arr0,Q1_slide_pred_arr0,Q3_slide_pred_arr0))


		### statistics on slide1 predictions ###
		print('### statistics on slide1 predictions ###')
		mean_slide_pred_arr1 = np.mean(slide_pred_arr1)
		std_slide_pred_arr1 = np.std(slide_pred_arr1)
		median_slide_pred_arr1, Q1_slide_pred_arr1, Q3_slide_pred_arr1 = np.percentile(slide_pred_arr1, (50,25,75), interpolation='nearest')
		print('mean_slide_pred_arr1: {:.3f}, std_slide_pred_arr1: {:.3f}, median_slide_pred_arr1: {:.3f}, Q1_slide_pred_arr1: {:.3f}, Q3_slide_pred_arr1: {:.3f}'.format(mean_slide_pred_arr1,std_slide_pred_arr1,median_slide_pred_arr1,Q1_slide_pred_arr1,Q3_slide_pred_arr1))
		with open(out_file, 'a') as f_out_file:
			f_out_file.write('### statistics on slide1 predictions ###\n')
			f_out_file.write('# mean\tstd\tmedian\tQ1\tQ3\n')
			f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(mean_slide_pred_arr1,std_slide_pred_arr1,median_slide_pred_arr1,Q1_slide_pred_arr1,Q3_slide_pred_arr1))
		

		##### paired sample wilcoxon test #####
		print('##### paired sample wilcoxon test #####')
		result = wilcoxon(slide_pred_arr0, slide_pred_arr1)
		wilcoxon_p_val_list.append(result.pvalue)
		print('wilcoxon test: statistic={:.3f}, p-value={:.1e}'.format(result.statistic, result.pvalue))
		with open(out_file, 'a') as f_out_file:
			f_out_file.write('##### paired sample wilcoxon test #####\n')
			f_out_file.write('# wilcoxon test: statistic={:.3f}, p-value={:.1e}\n'.format(result.statistic, result.pvalue))
			f_out_file.write('# mean_slide0\tstd_slide0\tmedian_slide0\tQ1_slide0\tQ3_slide0\tmean_slide1\tstd_slide1\tmedian_slide1\tQ1_slide1\tQ3_slide1\tpvalue\n')
			f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1e}\n'.format(mean_slide_pred_arr0,std_slide_pred_arr0,median_slide_pred_arr0,Q1_slide_pred_arr0,Q3_slide_pred_arr0,mean_slide_pred_arr1,std_slide_pred_arr1,median_slide_pred_arr1,Q1_slide_pred_arr1,Q3_slide_pred_arr1,result.pvalue))


		with open(out_file2, 'a') as f_out_file2:
			f_out_file2.write('{}\t{:.1e}\n'.format(temp_sample_id,wilcoxon_p_val_list[-1]))


		##### plot difference histogram #####

		difference =  slide_pred_arr0 - slide_pred_arr1

		fig1, ax1 = plt.subplots(figsize=(3,3))
		n, bins, patches = ax1.hist(difference, 20, density=False, facecolor='black', alpha=0.75)
		ax1.set_xlabel('difference')
		ax1.set_ylabel('# samples')
		ax1.set_title('Wilcoxon test: P={:.1e}'.format(result.pvalue))
		ax1.set_axisbelow(True)
		ax1.grid()

		fig1.tight_layout()
		fig_filename1 = '{}/difference_slide0_slide1__histogram__{}.pdf'.format(data_folder_path_slides_processed_sample,temp_sample_id)
		fig1.savefig(fig_filename1, dpi=200)

		plt.close('all')
		# plt.show()


##### summary statistics for statistical tests pvalues #####
print('##### summary statistics for statistical tests pvalues #####')

out_file3 = '{}/statistical_test__compare_top_bottom_slides_of_a_sample__pvalues__summary.txt'.format(data_folder_path_slides_processed)
with open(out_file3, 'w') as f_out_file3:
	f_out_file3.write('##### summary statistics for statistical tests pvalues #####\n')


### wilcoxon pvalues ###
print('### wilcoxon pvalues ###')
wilcoxon_p_val_arr = np.array(wilcoxon_p_val_list)
mean_wilcoxon_p_val_arr = np.mean(wilcoxon_p_val_arr)
std_wilcoxon_p_val_arr = np.std(wilcoxon_p_val_arr)
median_wilcoxon_p_val_arr, Q1_wilcoxon_p_val_arr, Q3_wilcoxon_p_val_arr = np.percentile(wilcoxon_p_val_arr, (50,25,75), interpolation='nearest')
print('mean_wilcoxon_p_val_arr: {:.1e}, std_wilcoxon_p_val_arr: {:.1e}, median_wilcoxon_p_val_arr: {:.1e}, Q1_wilcoxon_p_val_arr: {:.1e}, Q3_wilcoxon_p_val_arr: {:.1e}'.format(mean_wilcoxon_p_val_arr,std_wilcoxon_p_val_arr,median_wilcoxon_p_val_arr,Q1_wilcoxon_p_val_arr,Q3_wilcoxon_p_val_arr))
with open(out_file3, 'a') as f_out_file3:
	f_out_file3.write('### wilcoxon pvalues ###\n')
	f_out_file3.write('# mean\tstd\tmedian\tQ1\tQ3\n')
	f_out_file3.write('{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\t{:.1e}\n'.format(mean_wilcoxon_p_val_arr,std_wilcoxon_p_val_arr,median_wilcoxon_p_val_arr,Q1_wilcoxon_p_val_arr,Q3_wilcoxon_p_val_arr))


wilcoxon_data_list = [wilcoxon_p_val_arr]
num_datasets = 1

fig3, ax3 = plt.subplots(figsize=(3,3))
ax3.axhline(y=5e-2, linestyle='--', linewidth=1, color='k', alpha=0.5)

ax3.boxplot(wilcoxon_data_list, 
	widths=2., 
	sym='',
	whis=(5,95), 
	labels=['Wilcoxon test'], 
	positions=np.arange(num_datasets)*3+1, 
	patch_artist=True, 
	boxprops=dict(facecolor='black', color='black'),
	# capprops=dict(color=c),
	# whiskerprops=dict(color=c),
	# flierprops=dict(color=c, markeredgecolor=c),
	medianprops=dict(color='red'),
	)


ax3.set_xlim((-1,(num_datasets-1)*3+1+2))
ax3.set_yscale('log')
ax3.set_axisbelow(True)
ax3.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax3.set_ylabel('P-value (log-scale)',fontsize=10)
fig3.tight_layout()
fig3_filename = '{}/pvalues__boxplot.pdf'.format(data_folder_path_slides_processed)
fig3.savefig(fig3_filename, dpi=200)

plt.close('all')

# plt.show()



















