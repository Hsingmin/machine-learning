# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
#		numpy 1.13.1
#		tesseract 4.0
# -*- author: Hsingmin Lee
#
# icpr_image_slice.py -- Draw bounding boxes on images in dataset
# provided by Ali-ICPR MTWI 2018.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs

# Train images store path
TRAIN_IMAGE_DIR = 'd:/engineering-data/Ali-ICPR-data/train_image_9000'
# Train images bounding box path
TRAIN_BBOX_DIR = 'd:/engineering-data/Ali-ICPR-data/train_txt_9000'
# Train slices store path
TRAIN_SLICES_DIR = 'd:/engineering-data/Ali-ICPR-data/train_slice'

# Validate images store path
VALIDATE_IMAGE_DIR = 'd:/engineering-data/Ali-ICPR-data/validate_image_1000'
# Validate images bounding box path
VALIDATE_BBOX_DIR = 'd:/engineering-data/Ali-ICPR-data/validate_txt_1000'
# Validate slices store path
VALIDATE_SLICES_DIR = 'd:/engineering-data/Ali-ICPR-data/validate_slice'

# Preprocess image with given bounding-box as arguments, and get image slices .
#
# Preprocess image steps :
# 	create_images_list()
#	get_single_image_slice()
# 	get_single_image_bboxes()

# Get all images list and bounding box list
# arguments:
#	None
# returns:
#	image_list: list of full path for all images in training dataset
#	bbox_list: list of full path for all bounding boxes in training dataset
def create_image_list():
	image_list = []
	bbox_list = []

	image_dir = os.path.join(TRAIN_IMAGE_DIR, "T1F4yjFrVdXXXXXXXX_!!0-item_pic.jpg.jpg")
	bbox_dir = os.path.join(TRAIN_BBOX_DIR, "T1F4yjFrVdXXXXXXXX_!!0-item_pic.jpg.txt")
	image_list.append(image_dir)
	bbox_list.append(bbox_dir)

	return image_list, bbox_list

# Get single image bounding boxes 
# arguments: 
#	bbox_dir: bounding box file stored directory
# returns:
#	bboxes: a list of single image bounding boxes 
#	labels: a list of bounding box labels
def get_single_image_bboxes(bbox_dir):

	bbox_list = []
	labels = []
	with codecs.open(bbox_dir, 'r', 'utf-8') as bf:
		for line in bf:
			box = line.strip().split(',')
			labels.append(box[-1])
			bbox_list.append(box[:-1])

	'''
	sized_image = np.asarray(image_data.eval(session=sess), dtype='uint8')
	height = len(sized_image)
	width = len(sized_image[0])
	'''
	bboxes = []
	for bbox in bbox_list:
		bboxes.append([int(float(bbox[1])), int(float(bbox[0])), 
			int(float(bbox[5])), int(float(bbox[4]))])
	
	return bboxes, labels

# Get single image slice
# arguments:
#	image: a single decoded image data 
#	bboxes: bounding boxes list for single image
#	labels: labels corresponding to bbox in bboxes
# returns:
#	None
def get_single_image_slice(image, bboxes, labels):
	for i in range(len(bboxes)):
		bbox = bboxes[i]
		label = (labels[i]).replace('/', '').strip()
		if label == '###':
			continue
		
		path = "./to/" + label + ".jpg"
		begin = [bbox[0], bbox[1], 0]
		size = [bbox[2]-bbox[0], bbox[3]-bbox[1], -1]
		sliced_image = tf.slice(image, begin, size)
		#reshaped_image = tf.image.resize_images(sliced_image, [300, 300], method=0)
		reshaped_image = sliced_image
		uint8_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.uint8)
		encoded_image = tf.image.encode_jpeg(uint8_image)
		with tf.gfile.GFile(path, "wb") as f:
			f.write(encoded_image.eval())



def main(argv=None):
	# Get images list and corresponding boxes list.
	image_list, bbox_list = create_image_list()
	with tf.Session() as sess:
		for i in range(len(image_list)):
			image_dir = image_list[i]
			bbox_dir = bbox_list[i]
			image_raw_data = tf.gfile.FastGFile(image_dir, "rb").read()
			image_data = tf.image.decode_jpeg(image_raw_data)
			bboxes, labels = get_single_image_bboxes(bbox_dir)
			get_single_image_slice(image_data, bboxes, labels)


if __name__ == '__main__':
	main(argv=None)































