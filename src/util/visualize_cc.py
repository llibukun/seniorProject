# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Editor since: Tuesday, June 4th, 2019

# Author: Yutai Zhou
# visualize_cc.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_connected_components(img_labeled):
	"""
	This function takes a binary image, find its connected components 
	and converts each connected group into a different RGB color 
	to better visualize the results.
	"""

	# map component labels to a hue value
	MAX_VAL = 255
	label_hue = np.uint8(MAX_VAL*img_labeled/np.max(img_labeled))
	blank_ch = MAX_VAL*np.ones_like(label_hue) # white screen?
	img_ccomp = cv2.merge([label_hue, blank_ch, blank_ch])

	# convert the RGB
	img_ccomp = cv2.cvtColor(img_ccomp, cv2.COLOR_BGR2HSV)

	# set background to black
	img_ccomp[label_hue == 0] = 0

	return img_ccomp