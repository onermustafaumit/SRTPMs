import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image
import imageio

parser = argparse.ArgumentParser(description='')

parser.add_argument('--data_folder_path', default='', help='data folder path', dest='data_folder_path')
parser.add_argument('--tissue_mask_dir', default='../Images/primary_solid_tumor_tissue_masks_level6', help='Image directory', dest='tissue_mask_dir')
parser.add_argument('--normal_tissue_mask_dir', default='../Images/solid_tissue_normal_tissue_masks_level6', help='Image directory', dest='normal_tissue_mask_dir')

FLAGS = parser.parse_args()

data_folder_path = FLAGS.data_folder_path

slide_ids_list = [d for d in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, d))]
num_slides = len(slide_ids_list)
print("Num_slides: {}".format(num_slides))

slide_ids_list = sorted(slide_ids_list)
for i,slide_id in enumerate(slide_ids_list):

	sample_id = slide_id[:15]
	patient_id = slide_id[:12]


	if sample_id[-2:] =='01':
		tissue_mask_dir = FLAGS.tissue_mask_dir
	else:
		tissue_mask_dir = FLAGS.normal_tissue_mask_dir


	print('Slide {}/{}: {}'.format(i+1,num_slides,slide_id))

	slide_data_folder_path = '{}/{}'.format(data_folder_path,slide_id)
	data_file = '{}/patch_scores_{}.txt'.format(slide_data_folder_path,slide_id)
	patch_scores_data = np.loadtxt(data_file, delimiter='\t', comments='#', dtype=str)
	coordinates_arr = np.asarray(patch_scores_data[:,:2], dtype=int)
	patch_scores_arr = np.asarray(patch_scores_data[:,2], dtype=np.float32)
	patch_scores_arr = np.clip(patch_scores_arr, 0, 1)

	# get rgb image
	rgb_img_path = '{}/{}_rgb_image_level6.jpeg'.format(tissue_mask_dir,slide_id)
	rgb_img = Image.open(rgb_img_path).convert("RGB")
	rgb_img_arr = np.asarray(rgb_img, dtype=np.uint8)
	img_size = rgb_img_arr.shape

	cmap1 = cm.brg
	norm1 = mpl.colors.Normalize(vmin=0, vmax=1)
	m1 = cm.ScalarMappable(norm=norm1, cmap=cmap1)

	cmap2 = cm.viridis
	norm2 = mpl.colors.Normalize(vmin=np.amin(patch_scores_arr), vmax=np.amax(patch_scores_arr))
	m2 = cm.ScalarMappable(norm=norm2, cmap=cmap2)

	purity_map_arr = np.zeros(img_size[:2],dtype=np.float32)
	purity_map_color_arr = np.zeros((img_size[0],img_size[1],4),dtype=np.float32)
	purity_map_color_relative_arr = np.zeros((img_size[0],img_size[1],4),dtype=np.float32)
	for j in range(coordinates_arr.shape[0]):
		temp_row_id = coordinates_arr[j][0]
		temp_col_id = coordinates_arr[j][1]
		temp_score = patch_scores_arr[j]
		purity_map_arr[temp_row_id:temp_row_id+16,temp_col_id:temp_col_id+16] = temp_score

		temp_color1 = m1.to_rgba(temp_score)
		purity_map_color_arr[temp_row_id:temp_row_id+16,temp_col_id:temp_col_id+16,:] = temp_color1

		temp_color2 = m2.to_rgba(temp_score)
		purity_map_color_relative_arr[temp_row_id:temp_row_id+16,temp_col_id:temp_col_id+16,:] = temp_color2


	figure_size = ((3/img_size[0])*img_size[1]*1.05 + 0.1 + 0.6,3)
	right_val = 1 - 0.6/figure_size[0]
	# print(figure_size)

	# Plot the purity map - absolute
	fig1, ax1 = plt.subplots(figsize=figure_size)
	im1 = ax1.imshow(purity_map_color_arr)
	ax1.set_xticks([])
	ax1.set_yticks([])

	# Create colorbar
	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	divider1 = make_axes_locatable(ax1)
	cax1 = divider1.append_axes("right", size="5%", pad=0.1)
	cbar1 = ax1.figure.colorbar(m1, cax=cax1)
	cbar1.ax.set_ylabel("purity", rotation=-90, va="bottom")

	fig1.tight_layout()
	fig1.subplots_adjust(left=0.01, bottom=0.02,right=right_val, top=0.98, wspace=0.20, hspace=0.20)
	fig1_filename = '{}/{}__purity_map_color_with_colorbar.pdf'.format(slide_data_folder_path,slide_id)
	fig1.savefig(fig1_filename, dpi=200)


	# Plot the purity map - relative
	fig2, ax2 = plt.subplots(figsize=figure_size)
	im2 = ax2.imshow(purity_map_color_relative_arr)
	ax2.set_xticks([])
	ax2.set_yticks([])

	# Create colorbar
	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	divider2 = make_axes_locatable(ax2)
	cax2 = divider2.append_axes("right", size="5%", pad=0.1)
	cbar2 = ax2.figure.colorbar(m2,cax=cax2)
	cbar2.ax.set_ylabel("purity", rotation=-90, va="bottom")

	fig2.tight_layout()
	fig2.subplots_adjust(left=0.01, bottom=0.02,right=right_val, top=0.98, wspace=0.20, hspace=0.20)
	fig2_filename = '{}/{}__purity_map_color_relative_with_colorbar.pdf'.format(slide_data_folder_path,slide_id)
	fig2.savefig(fig2_filename, dpi=200)

	# plt.show()
	plt.close('all')


	# rgb image
	rgb_img_file = '{}/{}__rgb_img.jpeg'.format(slide_data_folder_path,slide_id)			
	imageio.imwrite(rgb_img_file, rgb_img_arr)

	# grayscale purity map
	purity_map_arr = np.asarray(purity_map_arr*255, dtype=np.uint8)
	purity_map_file = '{}/{}__purity_map.png'.format(slide_data_folder_path,slide_id)			
	imageio.imwrite(purity_map_file, purity_map_arr)

	# absolute purity map
	purity_map_color_arr = np.asarray(purity_map_color_arr*255, dtype=np.uint8)
	purity_map_color_file = '{}/{}__purity_map_color.png'.format(slide_data_folder_path,slide_id)			
	imageio.imwrite(purity_map_color_file, purity_map_color_arr)

	# relative purity map
	purity_map_color_relative_arr = np.asarray(purity_map_color_relative_arr*255, dtype=np.uint8)
	purity_map_color_relative_file = '{}/{}__purity_map_color_relative.png'.format(slide_data_folder_path,slide_id)			
	imageio.imwrite(purity_map_color_relative_file, purity_map_color_relative_arr)



