
# ocpr_image_preprocess.py -- Preprocess the Image .

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

INPUT_DATA = 'D:/engineering-data/Ali-ICPR-data/train_slice'
ALLIGNED_DATA = 'D:/engineering-data/Ali-ICPR-data/alligned_slice'
ALLIGNED_HEIGHT = 64
ALLIGNED_WIDTH = 256
NNI = 1

# Adjust image color . Define different order to adjust brightness, constrast,
# hue, saturation and whitening , which may affect the result .
# Programmes can pick one order randomly to reduce the influence on model .
def distort_color(image, color_ordering=0):
	if color_ordering == 0:
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
	elif color_ordering == 1:
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	elif color_ordering == 2:
		image = tf.image.random_hue(image, max_delta=0.2)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
	else:
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)

	# Normalize value in image tensor . 
	return tf.clip_by_value(image, 0.0, 1.0)

# Get all slices filename.
def get_images(path):
    images = []
    for rootdir, subdirs, files in os.walk(path):
        for file in files:
            images.append(os.path.join(rootdir, file))
    return images

def main(argv=None):

    image_list = get_images(INPUT_DATA)
    with tf.Session() as sess:
        for image in image_list:
            # Get image raw data in bytes type .
            image_raw_data = tf.gfile.GFile(image, "rb").read()
            img_data = tf.image.decode_png(image_raw_data)
            label = os.path.basename(image).split('.')[0]
            print("Preprocessing %s ============== " %(label))

            # Resize image into (height=64, width=256) 
            # with Nearest Neighbour Interpolation Algorithm. 
            image = tf.image.resize_images(img_data, [ALLIGNED_HEIGHT, ALLIGNED_WIDTH], method=NNI)
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

            resized = np.asarray(image.eval(), dtype='uint8')
            resized_image = Image.fromarray(resized)
            try:
                resized_image.save(os.path.join(ALLIGNED_DATA, label + '.png'))
            except Exception as e:
                print(e)
                pass
if __name__ == '__main__':
    tf.app.run()





























