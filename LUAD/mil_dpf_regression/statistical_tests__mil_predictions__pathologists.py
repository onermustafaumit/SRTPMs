import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, wilcoxon
import math

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


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='')
FLAGS = parser.parse_args()

fig1, ax1 = plt.subplots(figsize=(3,3))

out_file = '{}/statistical_tests__mil__pathologists__summary.txt'.format(FLAGS.data_folder_path)
with open(out_file, 'w') as f_out_file:
	f_out_file.write('# Statistical tests to compare mil predictions vs. percent tumor nuclei estimates\n')

data_file = '{}/sample_id__percent_tumor_nuclei__purity__mil_pred.txt'.format(FLAGS.data_folder_path)
data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)

num_patients = data.shape[0]

y_data = np.asarray(data[:,-2], dtype=float) # genomic tumor purity
x1_data = np.asarray(data[:,-1], dtype=float) # mil predictions
x2_data = np.asarray(data[:,-3], dtype=float) # percent tumor nuclei estimates


##### comparing correlated correlation coefficients using method of Meng et al. 1991 #####
print('##### comparing correlated correlation coefficients using method of Meng et al. 1991 #####')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('##### comparing correlated correlation coefficients using method of Meng et al. 1991 #####\n')

## x1 - y ##
print('## genomic tumor purity - mil predictions ##')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('## genomic tumor purity - mil predictions ##\n')

abs_error1 = np.abs(y_data - x1_data)

rho1, pval1 = stats.spearmanr(x1_data,y_data)

sorted_rho, rho_lower1, rho_upper1 = BootStrap(x1_data, y_data, n_bootstraps=2000)

print('rho1:{:.3f} (95% CI: {:.3f} - {:.3f}; P={:.1e})'.format(rho1,rho_lower1,rho_upper1,pval1))

with open(out_file, 'a') as f_out_file:
	f_out_file.write('rho1:{:.3f} (95% CI: {:.3f} - {:.3f}; P={:.1e})\n'.format(rho1,rho_lower1,rho_upper1,pval1))


ax1.hist(sorted_rho, 20, density=True, facecolor='g', alpha=0.75, label='MIL prediction')


## x2 - y ##
print('## genomic tumor purity - percent tumor nuclei ##')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('## genomic tumor purity - percent tumor nuclei ##\n')

abs_error2 = np.abs(y_data - x2_data)

rho2, pval2 = stats.spearmanr(x2_data,y_data)

sorted_rho, rho_lower2, rho_upper2 = BootStrap(x2_data, y_data, n_bootstraps=2000)

print('rho2:{:.3f} (95% CI: {:.3f} - {:.3f}; P={:.1e})'.format(rho2,rho_lower2,rho_upper2,pval2))

with open(out_file, 'a') as f_out_file:
	f_out_file.write('rho2:{:.3f} (95% CI: {:.3f} - {:.3f}; P={:.1e})\n'.format(rho2,rho_lower2,rho_upper2,pval2))

ax1.hist(sorted_rho, 20, density=True, facecolor='r', alpha=0.75, label='Percent tumor nuclei')


ax1.legend()
ax1.set_axisbelow(True)
ax1.grid()
ax1.set_title(r'Spearman $\rho$')
fig1.tight_layout()
fig_filename = '{}/spearman_rho_histogram_bootstrapping__mil__pathologists.pdf'.format(FLAGS.data_folder_path)
fig1.savefig(fig_filename, dpi=200)


## x1 - x2 ##
print('## mil predictions - percent tumor nuclei ##')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('## mil predictions - percent tumor nuclei ##\n')

rho3, pval3 = stats.spearmanr(x1_data,x2_data)

sorted_rho, rho_lower3, rho_upper3 = BootStrap(x1_data,x2_data, n_bootstraps=2000)

print('rho3:{:.3f} (95% CI: {:.3f} - {:.3f}; P={:.1e})'.format(rho3,rho_lower3,rho_upper3,pval3))

with open(out_file, 'a') as f_out_file:
	f_out_file.write('rho3:{:.3f} (95% CI: {:.3f} - {:.3f}; P={:.1e})\n'.format(rho3,rho_lower3,rho_upper3,pval3))


## Meng at al. test #####
print('## Meng at al. test ##')

z1 = 0.5*math.log((1+rho1)/(1-rho1))
z2 = 0.5*math.log((1+rho2)/(1-rho2))

mean_rho_sq = (rho1**2 + rho2**2)/2

f = min((1-rho3)/(2*(1-mean_rho_sq)),1)

h = (1 - f*mean_rho_sq)/(1 - mean_rho_sq)

z_observed = (z1-z2)*math.sqrt( (num_patients-3)/(2*(1-rho3)*h) )

if z_observed > 0:
	p_val = 2*(1-norm.cdf(z_observed))
else:
	p_val = 2*(norm.cdf(z_observed))

print('z_observed:{:.3f}, p_val:{:.1e}'.format(z_observed,p_val))

with open(out_file, 'a') as f_out_file:
	f_out_file.write('## Meng at al. test ##\n')
	f_out_file.write('z_observed:{:.3f}, p_val:{:.1e}\n'.format(z_observed,p_val))
	f_out_file.write('## Statistical test summary on correlation coefficients ##\n')
	f_out_file.write('# rho1\tpval1\trho_lower1\trho_upper1\trho2\tpval2\trho_lower2\trho_upper2\tp_val\n')
	f_out_file.write('{:.3f}\t{:.1e}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1e}\t{:.3f}\t{:.3f}\t{:.1e}\n'.format(rho1,pval1,rho_lower1,rho_upper1,rho2,pval2,rho_lower2,rho_upper2,p_val))


##### comparing absolute errors using Wilcoxon signed-rank test #####
print('##### comparing absolute errors using Wilcoxon signed-rank test #####')

with open(out_file, 'a') as f_out_file:
	f_out_file.write('##### comparing absolute errors using Wilcoxon signed-rank test #####\n')


## genomic tumor purity - mil predictions ##
print('## genomic tumor purity - mil predictions ##')

mean_abs_error1 = np.mean(abs_error1)
std_abs_error1 = np.std(abs_error1)
median_abs_error1, Q1_abs_error1, Q3_abs_error1 = np.percentile(abs_error1, (50,25,75), interpolation='nearest')
print('mean_abs_error1: {:.3f}, std_abs_error1: {:.3f}, median_abs_error1: {:.3f}, Q1_abs_error1: {:.3f}, Q3_abs_error1: {:.3f}'.format(mean_abs_error1,std_abs_error1,median_abs_error1,Q1_abs_error1,Q3_abs_error1))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('## genomic tumor purity - mil predictions ##\n')
	f_out_file.write('mean_abs_error1: {:.3f}, std_abs_error1: {:.3f}, median_abs_error1: {:.3f}, Q1_abs_error1: {:.3f}, Q3_abs_error1: {:.3f}\n'.format(mean_abs_error1,std_abs_error1,median_abs_error1,Q1_abs_error1,Q3_abs_error1))
	

## genomic tumor purity - percent tumor nuclei estimates ##
print('## genomic tumor purity - percent tumor nuclei estimates ##')

mean_abs_error2 = np.mean(abs_error2)
std_abs_error2 = np.std(abs_error2)
median_abs_error2, Q1_abs_error2, Q3_abs_error2 = np.percentile(abs_error2, (50,25,75), interpolation='nearest')
print('mean_abs_error2: {:.3f}, std_abs_error2: {:.3f}, median_abs_error2: {:.3f}, Q1_abs_error2: {:.3f}, Q3_abs_error2: {:.3f}'.format(mean_abs_error2,std_abs_error2,median_abs_error2,Q1_abs_error2,Q3_abs_error2))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('## genomic tumor purity - percent tumor nuclei estimates ##\n')
	f_out_file.write('mean_abs_error2: {:.3f}, std_abs_error2: {:.3f}, median_abs_error2: {:.3f}, Q1_abs_error2: {:.3f}, Q3_abs_error2: {:.3f}\n'.format(mean_abs_error2,std_abs_error2,median_abs_error2,Q1_abs_error2,Q3_abs_error2))
	

# wilcoxon test
print('## Wilcoxon test ##')

result = wilcoxon(abs_error1, abs_error2)
print('statistic={:.3f}, p-value={:.1e}'.format(result.statistic, result.pvalue))
with open(out_file, 'a') as f_out_file:
	f_out_file.write('## Wilcoxon test ##\n')
	f_out_file.write('statistic={:.3f}, p-value={:.1e}\n'.format(result.statistic, result.pvalue))
	f_out_file.write('## Statistical test summary on absolute errors ##\n')
	f_out_file.write('# mean_abs_error1\tstd_abs_error1\tmedian_abs_error1\tQ1_abs_error1\tQ3_abs_error1\tmean_abs_error2\tstd_abs_error2\tmedian_abs_error2\tQ1_abs_error2\tQ3_abs_error2\tp_val\n')
	f_out_file.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1e}\n'.format(mean_abs_error1,std_abs_error1,median_abs_error1,Q1_abs_error1,Q3_abs_error1,mean_abs_error2,std_abs_error2,median_abs_error2,Q1_abs_error2,Q3_abs_error2,result.pvalue))


difference = abs_error1 - abs_error2

fig2, ax2 = plt.subplots(figsize=(3,3))

n, bins, patches = ax2.hist(difference, 20, density=False, facecolor='g', alpha=0.75)
ax2.set_xlabel('difference')
ax2.set_ylabel('# samples')
ax2.set_axisbelow(True)
ax2.grid()

fig2.tight_layout()
fig_filename = '{}/abs_error_difference_histogram_bootstrapping__mil__pathologists.pdf'.format(FLAGS.data_folder_path)
fig2.savefig(fig_filename, dpi=200)

# plt.show()
plt.close('all')
