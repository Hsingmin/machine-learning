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

# Create image list.
def get_single_image_bboxes(sess, image_dir, bbox_dir):
	image_raw_data = tf.gfile.FastGFile(image_dir, "r").read()
	image_data = tf.image.decode_jpeg(image_raw_data)
	bbox_list = []
	txt_list = []
	with codecs.open(bbox_dir, 'r', 'utf-8') as bf:
		for line in bf:
			box = line.strip().split(',')
			txt_list.append(box[-1])
			bbox_list.append(box[:-1])

	'''
	sized_image = np.asarray(image_data.eval(session=sess), dtype='uint8')
	height = len(sized_image)
	width = len(sized_image[0])
	'''

	# convert bbox into relative position .
	bboxes = []
	for bbox in bbox_list:
		bboxes.append([int(bbox[1]), int(bbox[0]), 
			int(bbox[5]), int(bbox[4])])
	
	return bboxes, txt_list

def get_single_image_slice(box, image, path):
	sliced_image = tf.slice(image, [box[0], box[1], 0], [box[2]-box[0], box[3]-box[1], -1])
	reshaped_image = tf.reshape(sliced_image, [300, 300, 3])
	
	uint8_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.uint8)
	encoded_image = tf.image.encode_jpeg(uint8_image)
	with tf.gfile.GFile(path, "wb") as f:
		f.write(encoded_image.eval())

	return reshaped_image

def creat_images_list():
	image_dirs = []
	bbox_dirs = []

	image_dir = os.path.join(TRAIN_IMAGE_DIR, "T1F4yjFrVdXXXXXXXX_!!0-item_pic.jpg.jpg")
	bbox_dir = os.path.join(TRAIN_BBOX_DIR, "T1F4yjFrVdXXXXXXXX_!!0-item_pic.jpg.txt")
	image_dirs.append(image_dir)
	bbox_dirs.append(bbox_dir)

	return image_dirs, bbox_dirs

def main(argv=None):
	# Get 

	with tf.Session() as sess:
































