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
# 	tf.image.decode_jpeg()
#	tf.image.resize_images()
# 	tf.sample_distorted_bounding_box()
#	tf.image.convert_image_dtype()
#	tf.expand_dims()
#	tf.image.draw_bounding_boxes()
#	tf.reshape()
#	tf.slice()
def get_image_slice(image, bbox):
	# Regard whole image as the attention part if bbox is none .
	if bbox is None:
		bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1,1,4])
		
	
	# Distort image randomly to reduce affect to model of noise .
	bbox_begin, bbox_size, draw_bbox  = tf.image.sample_distorted_bounding_box(tf.shape(image), 
			bounding_boxes=bbox, min_object_covered=0.1)

	# Convert image tensor data type .
	#
	# Image data type from uint8 to tf.float32 .
	# Expand dimensions from 3-D to 4-D .
	
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)	
	
	image = tf.expand_dims(image, 0)
	
	# tf.image.draw_bounding_boxes(arg1=image, arg2=draw_box)
	#
	# image : dimendion expanded 
	# draw_box : the 3rd result returned by tf.image.sample_distorted_bounding_box()
	distorted_image = tf.image.draw_bounding_boxes(image, draw_bbox)	
	
	distorted_image = tf.reshape(distorted_image, [height, width, 3])
	
	distorted_image = tf.slice(distorted_image, bbox_begin, bbox_size)
	
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	distorted_image = distort_color(distorted_image, np.random.randint(5))

	return distorted_image

# Create image list.
def create_image_list(sess):
	image_dir = os.path.join(TRAIN_IMAGE_DIR, "T1F4yjFrVdXXXXXXXX_!!0-item_pic.jpg.jpg")
	bbox_dir = os.path.join(TRAIN_BBOX_DIR, "T1F4yjFrVdXXXXXXXX_!!0-item_pic.jpg.txt")
	image_raw_data = tf.gfile.FastGFile(image_dir, "r").read()
	image_data = tf.image.decode_jpeg(image_raw_data)
	bbox_list = []
	txt_list = []
	with codecs.open(bbox_dir, 'r', 'utf-8') as bf:
		for line in bf:
			box = line.strip().split(',')
			txt_list.append(box[-1])
			bbox_list.append(box[:-1])

	sized_image = np.asarray(image_data.eval(session=sess), dtype='uint8')
	height = len(sized_image)
	width = len(sized_image[0])
	
	# convert bbox into relative position .
	for bbox in bbox_list:
		bbox = [int(b/(int))]
	
def main(argv=None):
	# Get 































