# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Created: Tuesday, July 2nd, 2019
# extract_features.py

import os, cv2, time
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import thin
from util import crop_borders, visualize_connected_components, compute_main_axis_length
from tabulate import tabulate

# program start time
start_time = time.time()

# get the input images
image_folder = 'data/images' # folder path
working_dir = osp.dirname(__file__) # current working directory
image_dir = osp.join(working_dir, image_folder)
images = os.listdir(image_dir) # input image names

# save image features
img_features = []

# process each image
for img in images:
	# image start time
	img_start_time = time.time()

	# load image in color, grayscale
	img_path = osp.join(image_dir, img) # full image path
	img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # load image in grayscale
	
	# convert gray image to binary, invert, crop borders 
	THRESH = 0 # dummy threshold value since OTSU re-adjusts the threshold
	MAX_VAL = 255 # 8-bit max color value
	_, img_bw = cv2.threshold(img_gray, THRESH, MAX_VAL, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	img_invert = cv2.bitwise_not(img_bw) # invert the image
	img_cropped = crop_borders(img_invert) # crop the image
	
	# *********** DENSITY *********************************************************
	# closing to fill in gaps of panicle
	k_size = 180
	kernel_cl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
	img_close = cv2.morphologyEx(img_cropped, cv2.MORPH_CLOSE, kernel_cl)

	# find contours on current image
	_, contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# create a convex hull for each contour
	convex_hulls = []
	for contour in contours:
		convex_hulls.append(cv2.convexHull(contour, False))
	
	# visualization - draw the convex hulls
	num_channels = 3 # number of channels
	gold = (255, 215, 0) # RGB gold convex hulls
	dark_green = (0, 100, 0) # RGB dark green color
	convexhull_img = np.zeros((img_close.shape[0], img_close.shape[1], num_channels), np.uint8)
	cv2.drawContours(convexhull_img, contours, -1, dark_green, 1, 8, hierarchy)
	cv2.drawContours(convexhull_img, convex_hulls, -1, gold, 1, 8)

	# record elapsed time
	convex_start_time = time.time()

	# find the area of the largest convex hull
	area_largest_hull = 0
	large_index = 0
	for index, convex_hull in enumerate(convex_hulls):
		area_convex_hull = 0
		for row in range(img_cropped.shape[0]):
			for col in range(img_cropped.shape[1]):
				pt_location = cv2.pointPolygonTest(convex_hull, (row, col), False)
				if pt_location >= 0:
					area_convex_hull += 1
		if area_convex_hull > area_largest_hull:
			large_index = index
			area_largest_hull = area_convex_hull

	# report elapsed time
	convex_elapsed_time = time.time() - convex_start_time
	print(f'convex hull area elapsed time = {convex_elapsed_time:.2f}')

	# visualization - draw the largest convex hull
	tomato = (255, 99, 71) # RGB tomato convex hulls
	final_hull_img = np.zeros((img_close.shape[0], img_close.shape[1], num_channels), np.uint8)
	cv2.drawContours(final_hull_img, convex_hulls, large_index, tomato, 1, 8)

	# calculate the density = cropped image/closed image
	area_original = len(np.nonzero(img_cropped)[1])
	area_closing = len(np.nonzero(img_close)[1])
	density_morph = 100*area_original/area_closing
	density_hull = 100*area_original/area_largest_hull

	# *********** MAIN AXIS *******************************************************
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
	kernel_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
	img_open = cv2.morphologyEx(img_cropped, cv2.MORPH_OPEN, kernel_op)

	# connected component to find main axis, then find component containing main axis
	_, img_labels = cv2.connectedComponents(img_open)
	img_ccomp = visualize_connected_components(img_labels)

	# find the main axis skeleton
	_, counts = np.unique(img_labels, return_counts=True)
	second_most_common_label = np.argwhere(counts == np.sort(counts)[-2]).squeeze()
	binary_main_axis = img_labels == second_most_common_label
	img_main_axis = cv2.bitwise_and(img_cropped, img_cropped, mask = binary_main_axis.astype('uint8'))

	# calculate main axis diameter
	diameter_pixels = np.mean(binary_main_axis[-1:-40:-1,:]) * np.float64(binary_main_axis.shape[1])

	# calculate main axis length
	img_main_axis_thin = thin(binary_main_axis).astype('uint8') # first thin
	length_pixels, img_length = compute_main_axis_length(img_main_axis_thin) # then use dijkstra

	# *********** MAKE PLOTS ******************************************************
	image_figs = [
					(f'name: {img}    density_closing: {density_morph:.2f}    density_convexhull: {density_hull:.2f}    diameter_pixels: {diameter_pixels:.2f}    length_pixels: {length_pixels:.0f}', img_gray),  # original image
					# ('cropped', img_cropped), # cropped image
					# ('open main axis', img_open), # morphology opening
					# ('closed density', img_close), # morphology closing
					# ('convex hull density', convexhull_img), # convex hulls
					# ('largest hull density', final_hull_img), # largest convex hull
					# ('connected components', img_ccomp), # visualize_connected_components
					# ('main axis', img_main_axis), # the main axis
					# ('main axis skeleton', img_main_axis_thin), # skeleton
					# ('shortest path', img_length) # final length
				 ]

	for (title, image) in image_figs:
		fig1, ax1 = plt.subplots()
		ax1.imshow(image, cmap='gray')
		ax1.set_title(title)

	# *********** PRINT RESULTS ***************************************************
	print(f'{img}, Density: closing {density_morph:.2f} %, convex hull {density_hull:.2f} %, Diameter: {diameter_pixels:.2f}, Length: {length_pixels:.0f}\n')

	# save image features 
	img_info = []
	img_info.append(img) # column 1 - image name
	img_info.append(density_morph) # column 2 - image density (morph closing)
	img_info.append(density_hull) # column 3 - image density (convex hull)
	img_info.append(diameter_pixels) # column 4 - image main axis diameter
	img_info.append(length_pixels) # column 5 - image main axis length
	img_features.append(img_info) # all features of current image

	# image start time
	img_elapsed_time = time.time() - img_start_time

# *********** TABULATE *****************************************
img_header = ['image name', 'density closing', 'density convex hull', 'diameter pixels', 'length pixels']
print(tabulate(img_features[0:15], headers=img_header, tablefmt='fancy_grid', floatfmt='.2f'))
print(tabulate(img_features[15:30], headers=img_header, tablefmt='fancy_grid', floatfmt='.2f'))
print(tabulate(img_features[30:45], headers=img_header, tablefmt='fancy_grid', floatfmt='.2f'))

# report elapsed time
total_elapsed_time = time.time() - start_time
print(f'total elapsed time = {total_elapsed_time:.2f}')

# *******************************************************************************************

# *******************************************************************************************
# *******************************************************************************************
# ************ 2ND HALF --- PARTIAL COMPUTATIONS ********************************************
# # program start time
# start_time = time.time()

# # tabulate the data
# img_header = ['image name', 'density closing', 'density convex hull', 'diameter pixels', 'length pixels']
# print(tabulate(img_features[0:15], headers=img_header, tablefmt='fancy_grid', floatfmt='.2f'))
# print(tabulate(img_features[15:30], headers=img_header, tablefmt='fancy_grid', floatfmt='.2f'))
# print(tabulate(img_features[30:45], headers=img_header, tablefmt='fancy_grid', floatfmt='.2f'))

# # reload original images
# orig_imgs = []
# for img in images:
# 	# load image in color, grayscale
# 	img_gray = cv2.imread(osp.join(image_dir, img), cv2.IMREAD_GRAYSCALE) # load image in grayscale
# 	orig_imgs.append(img_gray)

# for index, img_info in enumerate(img_features):
# 	fig1, ax1 = plt.subplots()
# 	ax1.imshow(orig_imgs[index], cmap='gray')
#	ax1.set_title(f'name: {img_info[0]}    density_closing: {float(img_info[1]):.2f}    density_convexhull: {float(img_info[2]):.2f}    diameter_pixels: {float(img_info[3]):.2f}    length_pixels: {float(img_info[4]):.0f}')

# # report elapsed time
# total_elapsed_time = time.time() - start_time
# print(f'\n\ntotal elapsed time = {total_elapsed_time:.2f}')
