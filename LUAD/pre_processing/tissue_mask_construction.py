import numpy as np
import openslide
import cv2
import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--wsi_filelist', type=str, default='', help='The name of file containing the paths of WSIs.')
parser.add_argument('--out_dir', type=str, default='../Images', help='The directory where the cropped images will be stored.')
parser.add_argument('--tissue_type', type=str, default='primary_solid_tumor', help='primary_solid_tumor or solid_tissue_normal')
parser.add_argument('--mask_level', type=int, default=6, help='The level at which tissue masks will be prepared. level6: 16um/pixel')

FLAGS = parser.parse_args()

wsi_filelist = FLAGS.wsi_filelist
out_dir = os.path.join(FLAGS.out_dir,'{}_tissue_masks_level{}'.format(FLAGS.tissue_type,FLAGS.mask_level))

print('wsi_filelist:{}'.format(wsi_filelist))
print('out_dir:{}'.format(out_dir))

if not os.path.exists(out_dir):
	try:
		os.makedirs(out_dir)
	except:
		print("An exception occurred!")

target_res = 0.25*(2**FLAGS.mask_level)

wsi_count = 0
with open(wsi_filelist, 'r') as f_wsi_filelist:
	# next(f_wsi_filelist)
	for line in f_wsi_filelist:
		wsi_count += 1

		filepath = line.strip()
		# Skip lines starting with #
		if filepath[0] == '#':
			print('Skipped line!!! - WSI %d: %s' %(wsi_count,filepath))
			continue

		print('WSI {}:{}'.format(wsi_count,filepath))
		
		wsi_id = filepath.split('/')[-1].split('.')[0]

		slide = openslide.OpenSlide(filepath)

		val_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
		# print('Level0 Resolution:%3.2f' % val_x)

		if val_x < 0.3: # resolution:0.25um/pixel
			current_res = 0.25
		elif val_x < 0.6: # resolution:0.5um/pixel
			current_res = 0.5

		max_available_res = current_res * int(slide.level_downsamples[-1])

		im_read_level = slide.level_count -1
		im_size = slide.level_dimensions[im_read_level]
		try:
			im = slide.read_region((0, 0), im_read_level, im_size)
		except:
			print("An exception occurred!")
			continue
		
		im_arr = np.array(im)[:,:,[2,1,0]]


		im_resized_size = (im_size[0]//int(target_res/max_available_res),im_size[1]//int(target_res/max_available_res))
		# print('im_resized_size: {}'.format(im_resized_size))
		im_resized = cv2.resize(im_arr, im_resized_size)
		# print('im_resized shape:{}'.format(im_resized.shape))


		gray_im = cv2.cvtColor(im_resized, cv2.COLOR_RGB2GRAY)


		OTSU_thr, BW = cv2.threshold(gray_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# print('BW shape:{}'.format(BW.shape))

		inv_BW = cv2.bitwise_not(BW)

		kernel = np.ones((3,3), np.uint8)
		img_dilation = cv2.bitwise_not(cv2.dilate(inv_BW, kernel, iterations=1))

		BW_filtered = cv2.medianBlur(img_dilation,19)
		print('Percentage of background pixels:{:3.2f}%'.format( ((np.sum(BW_filtered)/255)/(im_resized_size[0]*im_resized_size[1]))*100 ) )

		des = cv2.bitwise_not(BW_filtered)
		contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contour:
			cv2.drawContours(des,[cnt],0,255,-1)
		img_filled = cv2.bitwise_not(des)
		
		inv_BW_filtered = cv2.bitwise_not(img_filled)

		b,g,r = cv2.split(im_resized)
		b = b & inv_BW_filtered
		g = g & inv_BW_filtered
		r = r & inv_BW_filtered
		im_out = cv2.merge((b,g,r))

		outfile = out_dir + '/' + wsi_id + '_rgb_image_level' + str(FLAGS.mask_level) + '.jpeg'
		cv2.imwrite( outfile, im_resized )
		outfile = out_dir + '/' + wsi_id + '_tissue_mask_level' + str(FLAGS.mask_level) + '.png'
		cv2.imwrite( outfile, img_filled )
		outfile = out_dir + '/' + wsi_id + '_masked_image_level' + str(FLAGS.mask_level) + '.jpeg'
		cv2.imwrite( outfile, im_out )

		# cv2.imshow('Original Image', im_arr)
		# cv2.imshow('Resized Image', im_resized)
		# cv2.imshow('Grayscale Image', gray_im)
		# cv2.imshow('Binary Image', BW)
		# cv2.imshow('Dilated Image', img_dilation)
		# cv2.imshow('Median filtered binary image', BW_filtered)
		# cv2.imshow('Filled binary image', img_filled)
		# cv2.imshow('Tissue Mask Image', im_out)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()


		# sys.exit()




