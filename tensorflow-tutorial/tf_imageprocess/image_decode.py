
# image_decode.py 

import matplotlib.pyplot as plt
import tensorflow as tf

# Get raw data of image .
image_raw_data = tf.gfile.FastGFile("./to/picture.jpg", 'rb').read()

# Convert image_data into uint8 tensor and write into file specified path .
def convert_write_image(reshaped, path):
	uint8_image = tf.image.convert_image_dtype(reshaped, dtype=tf.uint8)
	encoded_image = tf.image.encode_jpeg(uint8_image)
	with tf.gfile.GFile(path, "wb") as f:
		f.write(encoded_image.eval())

with tf.Session() as sess:
	
	# Get 3-dimension matrix by decoding picture in jpeg format .
	img_data = tf.image.decode_jpeg(image_raw_data)
	
	# print(img_data.eval())
	
	# Get iamge with pyplot tool .
	# plt.imshow(img_data.eval())
	# plt.show()

	# Convert data type into float32 .
	# img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
	# img_data = tf.cast(img_data, tf.float32)
	# Encode the img_data into a new image and save to file .
	# Get the same image as original .
	#
	# tf.image.encode_jpeg(image, ...)
	# image is a 3-dimension uint8 Tensor of shape [height, width, channels]

	'''
	# Reshape image decoded img_data to 300*300  
	# Argument method offers the reshape algorithm .
	resized = tf.image.resize_images(img_data, [512, 300], \
			method=tf.image.ResizeMethod.BILINEAR)
	convert_write_image(resized, "./to/resized_output.jpg")

	cropped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 700)
	convert_write_image(cropped, "./to/cropped_output.jpg")

	padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
	convert_write_image(padded, "./to/padded_output.jpg")
	
	central_cropped = tf.image.central_crop(img_data, 0.5)
	convert_write_image(central_cropped, "./to/central_cropped_output.jpg")
	
	# flipped = tf.image.random_flip_left_right(img_data)
	
	# tf.image.per_image_whitening() has been removed in higher version ,
	# use tf.image.per_image_standardization() 
	adjusted = tf.image.per_image_standardization(img_data)
	'''

	# Shrink the image to make bounding_box clear .
	img_data = tf.image.resize_images(img_data, [180, 267], method=1)

	# Cast image matrix data type into tf.float32 for tf.image.draw_bounding_boxes()
	# arguments needed to be real , the image data input should be a batch data , 
	# then expand it to be 4-D matrix .
	# batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)	

	# Bounding boxes relative position
	# bounding_boxes [y_min, x_min, y_max, x_max]
	boxes = tf.constant([[[0.05, 0.5, 0.9, 0.9], [0.35, 0.47, 0.5, 0.56]]])
	# result = tf.image.draw_bounding_boxes(batched, boxes)
	
	# ValueError: Tried to convert 'min_object_covered' to a tensor and failed .
	# Error: None values not supported .
	# 
	# Argument min_object_covered needed to be explicitly specified .
	begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(\
			tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.1)
	batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
	image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
	
	# reshaped = tf.reshape(result, [180, 267, 3])
	# reshaped = tf.reshape(image_with_box, [180, 267, 3])
	reshaped = tf.reshape(image_with_box, [180, 267, 3])
	
	reshaped = tf.slice(reshaped, begin, size)

	plt.imshow(reshaped.eval())
	plt.show()

'''
image_decoded = tf.image.decode_jpeg(tf.read_file('./to/picture.jpg'), channels=3)
cropped = tf.image.resize_image_with_crop_or_pad(image_decoded, 200, 200)
enc = tf.image.encode_jpeg(cropped)
fname = tf.constant('./to/image.jpg')
fwrite = tf.write_file(fname, enc)

sess = tf.Session()
result = sess.run(fwrite)
'''
























