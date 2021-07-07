import numpy as np

def calculate_sample_level_info(tcga_slide_data_path):

	# prepare tcga_sample
	tcga_slide_data_arr = np.loadtxt(tcga_slide_data_path, delimiter='\t', comments='#',dtype=str)
	if len(tcga_slide_data_arr) == 0:
		return
	if len(tcga_slide_data_arr.shape) == 1:
		tcga_slide_data_arr = tcga_slide_data_arr.reshape((1,-1))
		
	tcga_slide_id_arr = tcga_slide_data_arr[:,0]
	tcga_slide_id_set = set(list(tcga_slide_id_arr))

	analyte_portion_id_arr = np.asarray(tcga_slide_data_arr[:,1])
	percent_tumor_nuclei_arr = np.asarray(tcga_slide_data_arr[:,2])


	tcga_sample_dict = dict()
	tcga_sample_id_list = []
	for i, temp_slide_id in enumerate(tcga_slide_id_set):
		temp_sample_id = temp_slide_id[:16]

		tcga_slide_ind = np.where(tcga_slide_id_arr == temp_slide_id)[0][0]
		temp_analyte_portion_id = analyte_portion_id_arr[tcga_slide_ind]
		temp_percent_tumor_nuclei = percent_tumor_nuclei_arr[tcga_slide_ind]

		if temp_sample_id not in tcga_sample_dict:
			tcga_sample_id_list.append(temp_sample_id)
			tcga_sample_dict[temp_sample_id] = {'analyte_portion_id':[],'percent_tumor_nuclei':[]}

		tcga_sample_dict[temp_sample_id]['analyte_portion_id'].append(temp_analyte_portion_id)
		tcga_sample_dict[temp_sample_id]['percent_tumor_nuclei'].append(temp_percent_tumor_nuclei)

	tcga_sample_id_set = set(tcga_sample_id_list)

	folder_path = tcga_slide_data_path.split('/')[:-1]
	folder_path = '/'.join(folder_path)

	filename = tcga_slide_data_path.split('/')[-1]
	filename = filename.split('__')[1:]
	filename.insert(0,'sample_id')
	filename = '__'.join(filename)

	filepath = '{}/{}'.format(folder_path,filename)
	with open(filepath,'w') as f_filepath:
		f_filepath.write('# sample_id\tanalyte_portion_id\tpercent_tumor_nuclei\n')

		for i,temp_sample_id in enumerate(sorted(tcga_sample_id_set)):

			# print('{}: {}'.format(i+1,temp_sample_id))

			percent_tumor_nuclei_arr = np.asarray(tcga_sample_dict[temp_sample_id]['percent_tumor_nuclei'])

			analyte_portion_id = tcga_sample_dict[temp_sample_id]['analyte_portion_id'][0]
			
			try:
				percent_tumor_nuclei_arr = np.asarray(tcga_sample_dict[temp_sample_id]['percent_tumor_nuclei'],dtype=float)
				percent_tumor_nuclei = "{:.1f}".format(np.mean(percent_tumor_nuclei_arr))
			except:
				percent_tumor_nuclei = "None"


			f_filepath.write('{}\t{}\t{}\n'.format(temp_sample_id,analyte_portion_id,percent_tumor_nuclei))

	if len(tcga_sample_id_set) == 1:
		return

