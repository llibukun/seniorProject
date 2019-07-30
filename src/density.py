# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Editor since: Sunday, June 2nd, 2019

# Author: Yutai Zhou
# density.py

# for python 2 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import os, cv2
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from util import crop_borders # sub-functions

# get the input images
image_folder = 'data/select_images' # folder path
working_dir = osp.dirname(__file__) # current working directory
image_dir = osp.join(working_dir, image_folder)
images = os.listdir(image_dir) # input image names

# save useful information
img_props = []

# process each image
for img in images:
	# save useful image 
	img_info = []
	img_info.append(img) # column 1 - image name

	# load image in color, grayscale
	img_path = osp.join(image_dir, img) # full image path
	img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # load image in grayscale
	
	# convert gray image to binary, invert, crop borders 
	THRESH = 0 # dummy threshold value since OTSU re-adjusts the threshold
	MAX_VAL = 255 # 8-bit max color value
	_, img_bw = cv2.threshold(img_gray, THRESH, MAX_VAL, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	img_invert = cv2.bitwise_not(img_bw) # invert the image
	img_cropped = crop_borders(img_invert) # crop the image
	
	# closing to fill in gaps of panicle
	kernel_size = (180, 180)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
	img_close = cv2.morphologyEx(img_cropped, cv2.MORPH_CLOSE, kernel)
	
	# calculate the density = cropped/closed
	density = 100*np.sum(img_cropped)/np.sum(img_close)
	img_info.append(density) # column 2 - image density
	print(f'density = {density:.2f} %')

	# find contours on current image
	_, contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# create a new blank image
	num_channels = 3 # number of channels
	convexhull_img = np.zeros((img_close.shape[0], img_close.shape[1], num_channels), np.uint8)

	# create a convex hull for each contour
	convex_hulls = []
	for contour in contours:
		convex_hulls.append(cv2.convexHull(contour, False))
	
	# draw contours and convex hulls
	color_green = (0, 255, 0) # green contours
	color_blue = (0, 0, 255) # blue convex hulls
	for index in range(len(contours)):
		cv2.drawContours(convexhull_img, contours, index, color_green, 1, 8, hierarchy)
		cv2.drawContours(convexhull_img, convex_hulls, index, color_blue, 1, 8)

	# make plots
	image_figs = [
					('cropped', img_cropped), # cropped image
					('closed', img_close) # morphology closing
					('convex hull', convexhull_img) # convex hulls/contours
				 ]

	for (title, image) in image_figs:
		fig1, ax1 = plt.subplots()
		ax1.imshow(image, cmap='gray')
		ax1.set_title(title)

	img_props.append(img_info)

	# # count points belonging to CH
	# hull_index = int(input('Please enter index of the desired convex hull: '))
	# count = 0
	# for row in range(img_close.shape[0]):
	#     for col in range(img_close.shape[1]):
	#         pt_location = cv2.pointPolygonTest(convex_hulls[hull_index], (row, col), False)
	#         if pt_location >= 0:
	#             count += 1

	# print(f'Pixels in CH: {count}\nPixels in Panicle: {len(np.nonzero(img)[1])}\nRatio: {len(np.nonzero(img)[1])/count * 100:.4}%')
