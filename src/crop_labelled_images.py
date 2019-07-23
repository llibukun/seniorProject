# Layiwola Ibukun
# EGN4912: Switchgrass Root and Panicle Analysis
# Created: Friday, July 19th, 2019

# crop_labelled_images.py

import os, cv2
import os.path as osp
from util import crop_borders

# get the input images
image_folder = 'img_results' # input folder path
output_folder = 'polished_results' # output folder path
working_dir = osp.dirname(__file__) # current working directory
image_dir = osp.join(working_dir, image_folder)
images = os.listdir(image_dir) # input image names

# create the output directory
if not osp.exists(output_folder):
	os.makedirs(output_folder)

# process each image
for img in images:
	# load image in color, grayscale
	input_path = osp.join(image_dir, img) # full image path
	img_input = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) # load image in e# get the input images
	img_invert = cv2.bitwise_not(img_input) # invert the image
	img_cropped = crop_borders(img_invert) # crop the image
	reinvert_img = cv2.bitwise_not(img_cropped) # reinvert the image
	output_path = osp.join(working_dir, output_folder, img) # get cropped output image path
	cv2.imwrite(output_path, reinvert_img) # save image