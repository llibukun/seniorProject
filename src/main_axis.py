# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Editor since: Tuesday, June 25th, 2019

# Author: Yutai Zhou
# main_axis.py

import os 
import cv2
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from util import crop_borders, visualize_connected_components, compute_main_axis_length
from skimage.morphology import thin

# get the input images
image_folder = 'data/select_images' # folder path
working_dir = osp.dirname(__file__) # current working directory
image_dir = osp.join(working_dir, image_folder)
images = os.listdir(image_dir) # input image names

# process each image
for img in images:
	# load image in color, grayscale
	img_path = osp.join(image_dir, img) # full image path
	img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # load image in grayscale# get the input images

	# convert gray image to binary, invert, crop borders 
	THRESH = 0 # dummy threshold value since OTSU re-adjusts the threshold
	MAX_VAL = 255 # 8-bit max color value
	_, img_bw = cv2.threshold(img_gray, THRESH, MAX_VAL, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	img_invert = cv2.bitwise_not(img_bw) # invert the image
	img_cropped = crop_borders(img_invert) # crop the image
	
	# calculate erosion size with bottom 40 rows
	diams = []
	for row in img_cropped[-1:-40:-1,:]:
		diam = 0
		for num in row:
			if num != 0:
				diam += 1
			elif diam != 0:
				diams.append(diam)
				diam = 0

	# erosion size is the estimated diameter
	diam_np = np.array(diams)
	mean = np.mean(diam_np)
	median = np.median(diam_np)
	erosion_size = int(min(mean, median, 9))
	
	# morphology opening to remove non-main axis
	kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
	img_open = cv2.morphologyEx(img_cropped, cv2.MORPH_OPEN, kernel_op)

	# connected component to find main axis, then find component containing main axis
	_, img_labels = cv2.connectedComponents(img_open)
	img_ccomp = visualize_connected_components(img_labels)

	_, counts = np.unique(img_labels, return_counts=True)
	second_most_common_label = np.argwhere(counts == np.sort(counts)[-2]).squeeze()
	binary_main_axis = img_labels == second_most_common_label
	img_main_axis = cv2.bitwise_and(img_cropped, img_cropped, mask = binary_main_axis.astype('uint8'))

	# calculate main axis diameter
	diameter_pixels = np.mean(binary_main_axis[-1:-40:-1,:]) * np.float64(binary_main_axis.shape[1])

	# calculate main axis length
	# first thin
	img_main_axis_thin = thin(binary_main_axis).astype('uint8')

	# then use dijkstra
	length_pixels, img_length = compute_main_axis_length(img_main_axis_thin)

	print(f'{img}, Erosion size: {erosion_size}, Diameter: {diameter_pixels:.2f}, Length: {length_pixels:.0f}\n')

	# make plots
	image_figs = [
					# ('original', img_gray),  # original input
					('cropped', img_cropped), # cropped image
					# ('open', img_open), # morphology opening
					('connected components', img_ccomp), # visualize_connected_components
					# ('main axis', img_main_axis), # the main axis
					('main axis skeleton', img_main_axis_thin), # skeleton
					('shortest path', img_length)
				 ]

	for (title, image) in image_figs:
		fig1, ax1 = plt.subplots()
		ax1.imshow(image, cmap='gray')
		ax1.set_title(title)
