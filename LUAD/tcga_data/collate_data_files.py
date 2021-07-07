import numpy as np
import argparse

from slide_level_to_sample_level import calculate_sample_level_info

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_folder_path', default='.', help='')
FLAGS = parser.parse_args()

analyte_info_path = "{}/analyte_portion_submitter_ids.txt".format(FLAGS.data_folder_path)
slide_info_path = "{}/slide_ids__percent_tumor_nuclei_estimates.txt".format(FLAGS.data_folder_path)

# read slide_info
data = np.loadtxt(slide_info_path, delimiter='\t', skiprows=1, comments='#', dtype=str)
slide_id_arr = data[:,0]

slide_dict = dict()
for i in range(slide_id_arr.shape[0]):
	slide_id = slide_id_arr[i]
	sample_id = slide_id[:16]
	portion_id = slide_id[17:]
	sample_type = sample_id[-3:]

	# keep only slides of primary_solid_tumor samples with sample_type=='01A'
	if sample_type !='01A':
		continue

	# keep only slides with a valid percent_tumor_nuclei value
	try:
		percent_tumor_nuclei = float(data[i,-1])
	except:
		continue

	if sample_id not in slide_dict:
		slide_dict[sample_id] = []

	slide_dict[sample_id].append(portion_id)

# read analyte_info
analyte_id_arr = np.loadtxt(analyte_info_path, delimiter='\t', comments='#', dtype=str)

analyte_dict = dict()
for i in range(analyte_id_arr.shape[0]):
	analyte_id = analyte_id_arr[i]
	sample_id = analyte_id[:16]
	portion_id = analyte_id[17:]
	sample_type = sample_id[-3:]

	# keep only slides of primary_solid_tumor samples with sample_type=='01A'
	if sample_type !='01A':
		continue

	if sample_id not in analyte_dict:
		analyte_dict[sample_id] = []

	analyte_dict[sample_id].append(portion_id)




out_file = '{}/sample_id__slide_portion_ids__analyte_portion_ids.txt'.format(FLAGS.data_folder_path)
with open(out_file, 'w') as f_out_file:
	f_out_file.write('# sample_id\tslide_portion_ids\tanalyte_portion_ids\n')

	for key in sorted(slide_dict):

		if key not in analyte_dict:
			continue

		slide_portion_id_list = slide_dict[key]
		analyte_portion_id_list = analyte_dict[key]

		f_out_file.write('{}\t{}\t{}\n'.format(key,','.join(slide_portion_id_list),','.join(analyte_portion_id_list)))

		# print('{}\t{}\t{}'.format(key,','.join(slide_portion_id_list),','.join(analyte_portion_id_list)))


out_file = '{}/slide_id__analyte_portion_id__percent_tumor_nuclei.txt'.format(FLAGS.data_folder_path)
with open(out_file, 'w') as f_out_file:
	f_out_file.write('# slide_id\tanalyte_portion_id\tpercent_tumor_nuclei\n')


for i in range(slide_id_arr.shape[0]):
	slide_id = slide_id_arr[i]
	sample_id = slide_id[:16]
	sample_type = sample_id[-3:]

	# keep only slides of primary_solid_tumor samples with sample_type=='01A'
	if sample_type !='01A':
		continue

	# keep only slides with a valid percent_tumor_nuclei value
	try:
		percent_tumor_nuclei = float(data[i,-1])
	except:
		continue

	slide_portion_id_list = slide_dict[sample_id]
	num_slides = len(slide_portion_id_list)

	try:
		analyte_portion_id_list = analyte_dict[sample_id]
	except:
		continue

	# skip multiple-portion cases
	if len(analyte_portion_id_list) > 1:
		continue

	analyte_portion_id = analyte_portion_id_list[0]
	analyte_portion_type = analyte_portion_id[0]


	data_to_write = [slide_id,analyte_portion_id,str(percent_tumor_nuclei)]
	data_to_write = '\t'.join(data_to_write)

	with open(out_file, 'a') as f_out_file:
		f_out_file.write('{}\n'.format(data_to_write))

	

print('calculate_sample_level_info: {}'.format(out_file))
calculate_sample_level_info(out_file)













