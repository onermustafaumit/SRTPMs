import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, ttest_rel, normaltest, wilcoxon, mannwhitneyu
import math
import sys

plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})

def plot_sample__slide_preds(arr1,arr2,arr3,arr4,arr5):

	patient_id_arr = arr1
	patient_truth_arr = arr2
	patient_pred_arr = arr3
	slide_pred_arr0 = arr4
	slide_pred_arr1 = arr5

	# sort patient data based on truth values
	sorting_indices = np.argsort(patient_truth_arr)
	# sorting_indices = np.argsort(patient_pred_arr)
	patient_id_arr = patient_id_arr[sorting_indices]
	patient_truth_arr = patient_truth_arr[sorting_indices]
	patient_pred_arr = patient_pred_arr[sorting_indices]
	slide_pred_arr0 = slide_pred_arr0[sorting_indices]
	slide_pred_arr1 = slide_pred_arr1[sorting_indices]

	patient_id_list = []
	patient_pred_list = []
	patient_truth_list = []
	count = 0
	# fig, ax = plt.subplots(figsize=(14, 5))
	fig, ax = plt.subplots(figsize=(10, 5))
	for i in range(patient_id_arr.shape[0]):
		patient_id = patient_id_arr[i]
		patient_truth = patient_truth_arr[i]
		patient_pred = patient_pred_arr[i]
		slide_pred0 = slide_pred_arr0[i]
		slide_pred1 = slide_pred_arr1[i]


		count += 1

		patient_id_list.append(patient_id)
		patient_pred_list.append(patient_pred)
		patient_truth_list.append(patient_truth)

		if i == (patient_id_arr.shape[0]-1):
			ax.scatter(count,patient_truth, marker='x', color='r', label='genomic tumor purity')
			ax.scatter(count,patient_pred, marker='+', color='k', label='sample-level prediction')
			ax.scatter(count,slide_pred0, marker='.', color='k', label='slide-level prediction')
			ax.scatter(count,slide_pred1, marker='.', color='k')
		else:
			ax.scatter(count,patient_truth, marker='x', color='r')
			ax.scatter(count,patient_pred, marker='+', color='k')
			ax.scatter(count,slide_pred0, marker='.', color='k')
			ax.scatter(count,slide_pred1, marker='.', color='k')


		
	ax.set_ylim((-0.05,1.05))
	ax.set_yticks(np.arange(0,1.05,0.1))
	ax.set_xlim((0.5, len(patient_id_list) + 0.5))
	ax.set_xticks(np.arange(1,len(patient_id_list) + 0.5,1))
	ax.set_xticklabels(patient_id_list, rotation=90)
	# ax.set_xticks(patient_id_arr, rotation=90)
	ax.set_axisbelow(True)
	ax.grid(True)
	ax.legend()

	# ax.set_xlabel('predicted')
	ax.set_ylabel('tumor purity')

	fig.subplots_adjust(left=0.06, bottom=0.28, right=0.99, top=0.99, wspace=0.20 ,hspace=0.20 )
	# fig_filename = 'slide_and_patient_predictions_with_truth2_2__{}.png'.format(DATASET_TYPE)
	# # fig_filename = 'slide_and_patient_predictions2.png'
	# fig.savefig(fig_filename, dpi=200)

	return fig, ax


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='')
FLAGS = parser.parse_args()

data_folder_path_slides = FLAGS.data_folder_path
data_folder_path_samples = 'test_metrics/{}'.format('/'.join(data_folder_path_slides.split('/')[1:]))


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


# get slides predictions
sample_id_list = []
truth_list = []
sample_pred_list = []
slide_pred_list0 = []
slide_pred_list1 = []


data_file = '{}/slide_predictions_mpp.txt'.format(data_folder_path_slides)
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)
num_slides = data.shape[0]

for i in range(num_slides):
	temp_data = data[i]
	temp_sample_id = temp_data[0][:15]
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

	# first slide
	if i%2 == 0:
		sample_id_list.append(temp_sample_id)
		truth_list.append(sample_truth)
		sample_pred_list.append(sample_pred)
		slide_pred_list0.append(temp_pred)

	# second slide
	else:
		slide_pred_list1.append(temp_pred)


sample_id_arr = np.array(sample_id_list)
truth_arr = np.array(truth_list)
sample_pred_arr = np.array(sample_pred_list)
slide_pred_arr0 = np.array(slide_pred_list0)
slide_pred_arr1 = np.array(slide_pred_list1)

# plot sample and slide predictions 
fig, ax = plot_sample__slide_preds(sample_id_arr,truth_arr,sample_pred_arr,slide_pred_arr0,slide_pred_arr1)

fig_filename = '{}/sample_and_slide_predictions_with_genomic_tumor_purity_values.pdf'.format(data_folder_path_slides)
fig.savefig(fig_filename, dpi=200)


out_file = '{}/statistical_test__using_two_slides_vs_one_slide__summary.txt'.format(data_folder_path_slides)
with open(out_file, 'w') as f_out_file:
	f_out_file.write('##### slide-level prediction statistics and summary of statistical test #####\n')


##### abs_difference =  np.abs(slide_pred_arr0 - slide_pred_arr1) #####
print('### abs_difference =  np.abs(slide_pred_arr0 - slide_pred_arr1) ###')
with open(out_file, 'a') as f_out_file:
	f_out_file.write('### abs_difference =  np.abs(slide_pred_arr0 - slide_pred_arr1) ###\n')

abs_difference =  np.abs(slide_pred_arr0 - slide_pred_arr1)
mean_abs_difference = np.mean(abs_difference)
std_abs_difference = np.std(abs_difference)
median_abs_difference, Q1_abs_difference, Q3_abs_difference = np.percentile(abs_difference, (50,25,75), interpolation='nearest')
print('mean_abs_difference: {:.3f}, std_abs_difference: {:.3f}, median_abs_difference: {:.3f}, Q1_abs_difference: {:.3f}, Q3_abs_difference: {:.3f}'.format(mean_abs_difference,std_abs_difference,median_abs_difference,Q1_abs_difference,Q3_abs_difference))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('mean_abs_difference: {:.3f}, std_abs_difference: {:.3f}, median_abs_difference: {:.3f}, Q1_abs_difference: {:.3f}, Q3_abs_difference: {:.3f}\n'.format(mean_abs_difference,std_abs_difference,median_abs_difference,Q1_abs_difference,Q3_abs_difference))

fig0, ax0 = plt.subplots(figsize=(3,3))
n, bins, patches = ax0.hist(abs_difference, 20, density=False, facecolor='k', alpha=0.75)
ax0.set_xlabel(r'$d_{abs}$')
ax0.set_ylabel('# samples')
ax0.set_title(r'$m_{d_{abs}}$' + '={:.3f} (IQR: {:.3f} - {:.3f})'.format(median_abs_difference, Q1_abs_difference, Q3_abs_difference),fontsize=10)
ax0.set_axisbelow(True)
ax0.grid()

fig0.tight_layout()
fig_filename0 = '{}/abs_difference__between_slides_predictions__histogram.pdf'.format(data_folder_path_slides)
fig0.savefig(fig_filename0, dpi=200)


##### difference = abs_error_sample - mean_abs_error_slides #####
print('### difference = abs_error_sample - mean_abs_error_slides ###')
with open(out_file, 'a') as f_out_file:
	f_out_file.write('### abs_error_sample = np.abs(truth_arr - sample_pred_arr) ###\n')
	f_out_file.write('### mean_abs_error_slides = (abs_error0 + abs_error1)/2 ###\n')
	f_out_file.write('### difference = abs_error_sample - mean_abs_error_slides ###\n')

abs_error0 = np.abs(truth_arr - slide_pred_arr0)
abs_error1 = np.abs(truth_arr - slide_pred_arr1)
mean_abs_error_slides = (abs_error0 + abs_error1)/2
abs_error_sample = np.abs(truth_arr - sample_pred_arr)

mean_abs_error_sample = np.mean(abs_error_sample)
std_abs_error_sample = np.std(abs_error_sample)
median_abs_error_sample, Q1_abs_error_sample, Q3_abs_error_sample = np.percentile(abs_error_sample, (50,25,75), interpolation='nearest')
print('mean_abs_error_sample: {:.3f}, std_abs_error_sample: {:.3f}, median_abs_error_sample: {:.3f}, Q1_abs_error_sample: {:.3f}, Q3_abs_error_sample: {:.3f}'.format(mean_abs_error_sample,std_abs_error_sample,median_abs_error_sample,Q1_abs_error_sample,Q3_abs_error_sample))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('mean_abs_error_sample: {:.3f}, std_abs_error_sample: {:.3f}, median_abs_error_sample: {:.3f}, Q1_abs_error_sample: {:.3f}, Q3_abs_error_sample: {:.3f}\n'.format(mean_abs_error_sample,std_abs_error_sample,median_abs_error_sample,Q1_abs_error_sample,Q3_abs_error_sample))

mean_mean_abs_error_slides = np.mean(mean_abs_error_slides)
std_mean_abs_error_slides = np.std(mean_abs_error_slides)
median_mean_abs_error_slides, Q1_mean_abs_error_slides, Q3_mean_abs_error_slides = np.percentile(mean_abs_error_slides, (50,25,75), interpolation='nearest')
print('mean_mean_abs_error_slides: {:.3f}, std_mean_abs_error_slides: {:.3f}, median_mean_abs_error_slides: {:.3f}, Q1_mean_abs_error_slides: {:.3f}, Q3_mean_abs_error_slides: {:.3f}'.format(mean_mean_abs_error_slides,std_mean_abs_error_slides,median_mean_abs_error_slides,Q1_mean_abs_error_slides,Q3_mean_abs_error_slides))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('mean_mean_abs_error_slides: {:.3f}, std_mean_abs_error_slides: {:.3f}, median_mean_abs_error_slides: {:.3f}, Q1_mean_abs_error_slides: {:.3f}, Q3_mean_abs_error_slides: {:.3f}\n'.format(mean_mean_abs_error_slides,std_mean_abs_error_slides,median_mean_abs_error_slides,Q1_mean_abs_error_slides,Q3_mean_abs_error_slides))

difference = abs_error_sample - mean_abs_error_slides

#paired sample wilcoxon test
result = wilcoxon(abs_error_sample, mean_abs_error_slides)
print('wilcoxon test: statistic={:.3f}, p-value={:.1e}'.format(result.statistic, result.pvalue))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('wilcoxon test: statistic={:.3f}, p-value={:.1e}\n'.format(result.statistic, result.pvalue))
	f_out_file.write('# mean_abs_error_sample\tstd_abs_error_sample\tmedian_abs_error_sample\tQ1_abs_error_sample\tQ3_abs_error_sample\tmean_mean_abs_error_slides\tstd_mean_abs_error_slides\tmedian_mean_abs_error_slides\tQ1_mean_abs_error_slides\tQ3_mean_abs_error_slides\tpvalue\n')
	f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1e}\n'.format(mean_abs_error_sample,std_abs_error_sample,median_abs_error_sample,Q1_abs_error_sample,Q3_abs_error_sample,mean_mean_abs_error_slides,std_mean_abs_error_slides,median_mean_abs_error_slides,Q1_mean_abs_error_slides,Q3_mean_abs_error_slides,result.pvalue))


fig3, ax3 = plt.subplots(figsize=(3,3))

n, bins, patches = ax3.hist(difference, 20, density=False, facecolor='k', alpha=0.75)
# ax3.set_xlabel('difference')
ax3.set_xlabel(r'$e_{smpl} - e_{sld}$')
# ax3.set_xlabel(r'$|\hat{p}_{sample}-{p}_{sample}|-0.5*(|\hat{p}_{slide1}-{p}_{sample}|+|\hat{p}_{slide2}-{p}_{sample}|)$')
ax3.set_ylabel('# samples')
ax3.set_title('Wilcoxon test: P={:.1e}'.format(result.pvalue))
ax3.set_axisbelow(True)
ax3.grid()

fig3.tight_layout()
fig_filename3 = '{}/difference__abs_error_sample__mean_abs_error_slides__histogram.pdf'.format(data_folder_path_slides)
fig3.savefig(fig_filename3, dpi=200)

# plt.show()

plt.close('all')








