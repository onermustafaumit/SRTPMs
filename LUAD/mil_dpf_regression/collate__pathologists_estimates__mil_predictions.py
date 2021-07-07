import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='', help='data folder path', dest='data_folder_path')
FLAGS = parser.parse_args()

data_folder_path = FLAGS.data_folder_path

percent_tumor_nuclei_file = '../tcga_data/sample_id__analyte_portion_id__percent_tumor_nuclei.txt'
mil_predictions_file = '{}/patient_predictions_mpp.txt'.format(data_folder_path)


# get percent tumor nuclei data
percent_tumor_nuclei_data = np.loadtxt(percent_tumor_nuclei_file, delimiter='\t', comments='#', dtype=str)

percent_tumor_nuclei_dict = {}
for i in range(percent_tumor_nuclei_data.shape[0]):
	sample_id = percent_tumor_nuclei_data[i,0][:15]
	percent_tumor_nuclei = float(percent_tumor_nuclei_data[i,2])/100

	percent_tumor_nuclei_dict[sample_id] = percent_tumor_nuclei


out_file = '{}/sample_id__percent_tumor_nuclei__purity__mil_pred.txt'.format(data_folder_path)
with open(out_file,'w') as f_out_file:
	f_out_file.write('# sample_id\tpercent_tumor_nuclei\ttruth\tpred\n')

	# get mil predictions
	mil_predictions_data = np.loadtxt(mil_predictions_file, delimiter='\t', comments='#', dtype=str)
	for i in range(mil_predictions_data.shape[0]):
		sample_id = mil_predictions_data[i,0][:15]
		truth = float(mil_predictions_data[i,1])
		pred = float(mil_predictions_data[i,2])

		if sample_id in percent_tumor_nuclei_dict:
			percent_tumor_nuclei = percent_tumor_nuclei_dict[sample_id]

			f_out_file.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(sample_id,percent_tumor_nuclei,truth,pred))

