import os
import argparse
from PIL import Image
import numpy as np
import openslide
from openslide import open_slide
import fnmatch
from datetime import datetime
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

cwd = os.getcwd()

parser = argparse.ArgumentParser()

parser.add_argument('--patch_size', type=int, default=512, help='The size of the patches that will be cropped.')
parser.add_argument('--patch_level', type=int, default=1, help='The level from which patches will be cropped. level0:0.25um/pixel, level1:0.5um/pixel, etc.')
parser.add_argument('--mask_level', type=int, default=6, help='The level at which tissue masks were prepared. level6:16um/pixel')
parser.add_argument('--stride', type=int, default=512, help='stride - number of pixels at patch_level')
parser.add_argument('--patient_ids_list', type=str, default='', help='File containing the patient IDs')
parser.add_argument('--wsi_dir', type=str, default='../WSIs', help='')
parser.add_argument('--out_dir', type=str, default='../Images', help='The directory where the cropped patches will be stored.')
parser.add_argument('--tissue_type', type=str, default='primary_solid_tumor', help='primary_solid_tumor or solid_tissue_normal')

FLAGS = parser.parse_args()

patch_size = FLAGS.patch_size
patch_level_res = 0.25 * (2 ** FLAGS.patch_level)
mask_level_res = 0.25 * (2 ** FLAGS.mask_level)
resolution_ratio = mask_level_res / patch_level_res
mask_patch_size = int(patch_size / resolution_ratio)
mask_num_pixels = np.square(mask_patch_size)
stride_at_mask_level = int(FLAGS.stride/resolution_ratio)

out_dir = '{}/all_cropped_patches_{}__level{}__stride{}__size{}'.format(FLAGS.out_dir,FLAGS.tissue_type,FLAGS.patch_level,FLAGS.stride,patch_size)
tissue_mask_dir = '{}/{}_tissue_masks_level{}'.format(FLAGS.out_dir,FLAGS.tissue_type,FLAGS.mask_level)

print('out_dir: {}'.format(out_dir))
print('tissue_mask_dir: {}'.format(tissue_mask_dir))

if not os.path.exists(out_dir):
	try:
		os.makedirs(out_dir)
	except:
		print("An exception occurred!")

cropped_patches_info = out_dir + '/cropped_patches_info_' + FLAGS.patient_ids_list + '.txt'
# print(cropped_patches_info)

with open(cropped_patches_info, 'w') as f_cropped_patches_info:
	f_cropped_patches_info.write('# patient_id\tnumber_of_patches\n')

patient_ids_arr = np.loadtxt(FLAGS.patient_ids_list, dtype=str, comments='#', delimiter='\t')
patient_ids_arr = patient_ids_arr.reshape((-1,))
num_patient_ids = patient_ids_arr.shape[0]


for i in range(num_patient_ids):
	patient_id = patient_ids_arr[i]
	print('Patient-{}/{}: {}'.format(i+1,num_patient_ids,patient_id))

	num_cropped_patches = 0

	outdir = out_dir + '/' + patient_id
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	cropped_patches_filelist = outdir + '/cropped_patches_filelist.txt'
	# print(cropped_patches_filelist)

	with open(cropped_patches_filelist, 'w') as f_cropped_patches_filelist:
		f_cropped_patches_filelist.write('# patch_id\twsi_id\tmask_row\tmask_col\tbg_ratio\n')

	for root, dirnames, filenames in os.walk(tissue_mask_dir):
		for filename in fnmatch.filter(filenames, (patient_id + '-*_tissue_mask_level{}.png'.format(FLAGS.mask_level))):
			# print(filename)

			wsi_id = filename.split('_')[0]
			print('\t{}'.format(wsi_id))

			mask_img_path = root + '/' + filename
			mask_im_arr = np.array(Image.open(mask_img_path))[:,:]/255
			mask_height, mask_width = mask_im_arr.shape

			accepted_patch_mask = np.zeros((mask_height, mask_width),dtype=np.uint8)

			# plt.imshow(mask_im_arr, cmap='gray')
			# plt.show()

			for root2, dirnames2, filenames2 in os.walk(FLAGS.wsi_dir):
				for filename2 in fnmatch.filter(filenames2, (wsi_id + '*.svs')):
					slide_path = root2 + '/' + filename2

					slide = openslide.OpenSlide(slide_path)

					val_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
					# print('Level0 Resolution:%3.2f' % val_x)

					highest_res_level = round(val_x/0.25) - 1

					im_read_level = 0
					read_level_res = 0.25 * (2 ** highest_res_level)
					res_ratio_mask_to_read = mask_level_res / read_level_res
					res_ratio_patch_to_read = patch_level_res / read_level_res
					im_size = (int(patch_size * res_ratio_patch_to_read), int(patch_size * res_ratio_patch_to_read))

					mask_height_range = int((mask_height - mask_patch_size) / stride_at_mask_level) + 1
					mask_width_range = int((mask_width - mask_patch_size) / stride_at_mask_level) + 1
					# print(mask_height , mask_width)
					# print(mask_height_range , mask_width_range)

					pbar = tqdm(total=mask_height_range*mask_width_range)
					for row in range(mask_height_range):
						for col in range(mask_width_range):
							mask_patch_arr = mask_im_arr[row * stride_at_mask_level:row * stride_at_mask_level + mask_patch_size, col * stride_at_mask_level:col * stride_at_mask_level + mask_patch_size]
							# print(mask_patch_arr)

							pbar.update(1)

							if np.sum(mask_patch_arr) == 0:
								row_read = int(row * stride_at_mask_level * res_ratio_mask_to_read)
								col_read = int(col * stride_at_mask_level * res_ratio_mask_to_read)
								# print(row_read,col_read)
								
								im = slide.read_region((col_read, row_read), im_read_level, im_size)
								im_arr = np.array(im)[:, :, 0:3]
								# plt.imshow(im_arr)
								# plt.show()

								if res_ratio_patch_to_read > 1:
									im_arr = np.array(im.resize((patch_size, patch_size), Image.ANTIALIAS))[:, :, 0:3]

								im_arr_avg = np.mean(im_arr, axis=2)  # mean of RGB values combined per pixel
								im_arr_avg_bool = im_arr_avg > 240
								bg_ratio = np.sum(im_arr_avg_bool)/np.square(patch_size)
								# # print(im_arr_avg_bool)
								if bg_ratio > 0.75:
									# print(bg_ratio)
									# plt.imshow(im_arr)
									# plt.show()
									continue

								# plt.imshow(im_arr)
								# plt.show()

								accepted_patch_mask[row * stride_at_mask_level:row * stride_at_mask_level + mask_patch_size, col * stride_at_mask_level:col * stride_at_mask_level + mask_patch_size] = 255


								# outfile = outdir + '/' + str(num_cropped_patches) + '.png'
								outfile = outdir + '/' + str(num_cropped_patches) + '.jpeg'
								imageio.imwrite(outfile, im_arr)

								with open(cropped_patches_filelist, 'a') as f_cropped_patches_filelist:
									f_cropped_patches_filelist.write(str(num_cropped_patches) + '\t' + wsi_id + '\t' + str(row*stride_at_mask_level) + '\t' + str(col*stride_at_mask_level) + '\t' + str(bg_ratio) + '\n')

								num_cropped_patches += 1

					pbar.close()

			outfile = '{}/{}_accepted_patch_mask.png'.format(outdir,wsi_id)
			imageio.imwrite(outfile, accepted_patch_mask)

			# sys.exit()

	with open(cropped_patches_info, 'a') as f_cropped_patches_info:
		f_cropped_patches_info.write(str(patient_id) + '\t' + str(num_cropped_patches) + '\n')

