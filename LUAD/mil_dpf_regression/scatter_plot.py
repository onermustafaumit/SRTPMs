import numpy as np
import argparse
import os
import sys
from os import path

import matplotlib.pyplot as plt

from scipy import stats


def score_fnc(x_data, y_data):
	rho, pval = stats.spearmanr(x_data, y_data)
	return rho

def BootStrap(x_data, y_data, n_bootstraps):

	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []

	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(x_data), len(x_data))

		score = score_fnc(x_data[indices], y_data[indices])
		bootstrapped_scores.append(score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	
	return sorted_scores, confidence_lower, confidence_upper


plt.rcParams.update({'font.size':8, 'font.family':'Times New Roman'})

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='data folder path', dest='data_folder_path')
FLAGS = parser.parse_args()

data_folder_path = FLAGS.data_folder_path


##### MIL predictions #####

out_file = '{}/summary_spearmann_corr_coeff_and_abs_err__mil.txt'.format(FLAGS.data_folder_path)
with open(out_file, 'w') as f_out_file:
	f_out_file.write('# rho\trho_lower\trho_upper\tp_value\tmean_abs_err\tstd_abs_err\tmedian_abs_err\tQ1_abs_err\tQ3_abs_err\n')

# get data
data_file = '{}/patient_predictions_mpp.txt'.format(data_folder_path)
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)
temp_truth = np.asarray(data[:,1], dtype=float)
temp_prob = np.asarray(data[:,2], dtype=float)

rho, pval = stats.spearmanr(temp_prob,temp_truth)	
print('spearman rho:{:.3f} - pval:{:.1e}'.format(rho,pval))

sorted_rho, rho_lower, rho_upper = BootStrap(temp_prob, temp_truth, n_bootstraps=2000)
print('bootstrap: ({:.3f}-{:.3f})'.format(rho_lower,rho_upper))

# absolute error
abs_err = np.abs(temp_truth - temp_prob)
mean_abs_err = np.mean(abs_err)
std_abs_err = np.std(abs_err)
median_abs_err, Q1_abs_err, Q3_abs_err = np.percentile(abs_err, (50,25,75), interpolation='nearest')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.1e}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(rho,rho_lower,rho_upper,pval,mean_abs_err,std_abs_err,median_abs_err,Q1_abs_err,Q3_abs_err))


title_text = r'Spearman $\rho$ = {:.3f} (CI: {:.3f} - {:.3f})'.format(rho,rho_lower,rho_upper)

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
fig_filename = '{}/patient_level_scatter_plot__mil.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)

plt.show()


##### Pathologists' estimates #####

out_file = '{}/summary_spearmann_corr_coeff_and_abs_err__pathologists.txt'.format(FLAGS.data_folder_path)
with open(out_file, 'w') as f_out_file:
	f_out_file.write('# rho\trho_lower\trho_upper\tp_value\tmean_abs_err\tstd_abs_err\tmedian_abs_err\tQ1_abs_err\tQ3_abs_err\n')

# get data
data_file = '{}/sample_id__percent_tumor_nuclei__purity__mil_pred.txt'.format(data_folder_path)
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)
temp_truth = np.asarray(data[:,2], dtype=float)
temp_prob = np.asarray(data[:,1], dtype=float)

rho, pval = stats.spearmanr(temp_prob,temp_truth)	
print('spearman rho:{:.3f} - pval:{:.1e}'.format(rho,pval))

sorted_rho, rho_lower, rho_upper = BootStrap(temp_prob, temp_truth, n_bootstraps=2000)
print('bootstrap: ({:.3f}-{:.3f})'.format(rho_lower,rho_upper))

# absolute error
abs_err = np.abs(temp_truth - temp_prob)
mean_abs_err = np.mean(abs_err)
std_abs_err = np.std(abs_err)
median_abs_err, Q1_abs_err, Q3_abs_err = np.percentile(abs_err, (50,25,75), interpolation='nearest')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.1e}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(rho,rho_lower,rho_upper,pval,mean_abs_err,std_abs_err,median_abs_err,Q1_abs_err,Q3_abs_err))


title_text = r'Spearman $\rho$ = {:.3f} (CI: {:.3f} - {:.3f})'.format(rho,rho_lower,rho_upper)

fig, ax = plt.subplots(figsize=(3,3))

ax.plot([0,1], [0,1], linestyle='--', color='red', zorder=1)
ax.scatter(temp_prob,temp_truth, color='k', zorder=2, alpha=0.7, s=16)
ax.set_xlabel('Percent tumor nuclei')
ax.set_ylabel('Genomic tumor purity')
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
ax.set_axisbelow(True)
ax.grid()
ax.set_title(title_text)

fig.tight_layout()
fig_filename = '{}/patient_level_scatter_plot__pathologists.png'.format(data_folder_path)
fig.savefig(fig_filename, dpi=200)

plt.show()



