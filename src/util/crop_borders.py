# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Editor since: Sunday, June 2nd, 2019

# Author: Yutai Zhou
# crop_borders.py

import numpy as np

def crop_borders(img, margin=0.01, bg_thresh=0.006):
	"""
	This function crops away all unnnecesary background present in the input image to
	better calculate the plant density.
	
	Parameters:
	img - the input image
	margin - extra background margin as a percentage of the image size
	bg_thresh - the percentage of plant pixels that determine the borders
	"""

	# get image dimensions
	MAX_VAL = 255 # 8-bit max color value
	height = img.shape[0]
	width = img.shape[1]

	# calculate left-right border
	thresh_lr = np.ceil(bg_thresh*width).astype(int) # in pixels
	density_lr = np.sum(img, axis=0) >= (MAX_VAL*thresh_lr)
	borders_lr = np.argwhere(density_lr).squeeze()
	margin_lr = np.ceil(margin*width).astype(int) # in pixels

	# calculate top-bottom border
	thresh_tb = np.ceil(bg_thresh*height).astype(int) # in pixels
	density_tb = np.sum(img, axis=1) >= (MAX_VAL*thresh_tb)
	borders_tb = np.argwhere(density_tb).squeeze()
	margin_tb = np.ceil(margin*height).astype(int) # in pixels

	# set left-right border indices
	left = borders_lr[0] - margin_lr
	right = borders_lr[-1] + margin_lr

	# set top-bottom borders indices
	top = borders_tb[0] - margin_tb
	bottom = borders_tb[-1] + margin_tb

	# out of bounds check
	if left < 0:
		left = 0
	if right >= width:
		right = width - 1

	# out of bounds check
	if top < 0:
		top = 0
	if bottom >= width:
		bottom = width - 1

	# crop the image
	img_cropped = img[top:bottom, left:right]

	return img_cropped
