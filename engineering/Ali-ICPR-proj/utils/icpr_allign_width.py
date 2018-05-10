# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
#		numpy 1.13.1
#		tesseract 4.0
# -*- author: Hsingmin Lee
#
# icpr_image_tesseract.py -- Recognize text on image 
# sliced by icpr_image_slice.py .

import os
from PIL import Image

# Train slices store path
TRAIN_SLICES_DIR = 'd:/engineering-data/Ali-ICPR-data/train_slice'

TRUNCATED_LEN = 512

# Get train dataset and the labels.
# params:
#       path -- TRAIN_SLICES_DIR
# returns:
#       slice_list -- array holding slice file path
#       label_list -- array holding slice text
#
def get_slices(path):
	slice_list = []
	label_list = []

	for rootdir, subdirs, filenames in os.walk(path):
		for filename in filenames:
			slice_list.append(os.path.join(rootdir, filename))
			label_list.append(filename.replace('.jpg', '').replace('.png', ''))

	return slice_list, label_list

# Get slices width and label length.
# params:
#       path -- TRAIN_SLICES_DIR
# returns:
#       width_list -- array holding slices width
#       label_list -- array holding slices label
#
def get_slices_info(path):
    images, labels = get_slices(path)
    width_list = []
    label_list = []
    for i, image in enumerate(images):
        img = Image.open(image)
        width = img.size[0]
        if width <= TRUNCATED_LEN:
            width_list.append(width)
            label_list.append(labels[i])

    return width_list, label_list

if __name__ == '__main__':
    width_list, label_list = get_slices_info(TRAIN_SLICES_DIR)
    cnt = 0
    print('Max Width of Slices = %d' %(max(width_list)))
    print('Max Length of Labels = %d' %(max(len(label) for label in label_list)))

    for i, label in enumerate(label_list):
        if len(label) >= 20:
            cnt += 1
            print('%s -- %d' %(label, width_list[i]))
            print('Long label counts to = %d' %cnt)



















