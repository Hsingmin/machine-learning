import tensorflow as tf
import matplotlib.pyplot as plt

img_name = ["./to/beautiful.jpg"]
filename_queue = tf.train.string_input_producer(img_name)
img_reader = tf.WholeFileReader()
_,image_jpg = img_reader.read(filename_queue)

image_decode_jpeg = tf.image.decode_jpeg(image_jpg)
image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32)
img = image_decode_jpeg
image_decode_jpeg = tf.expand_dims(image_decode_jpeg, 0)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

box = tf.constant([[[0.2, 0.6, 0.5, 0.8]]])
image_bilinear = tf.image.draw_bounding_boxes(sess.run(image_decode_jpeg), box)

image_bilinear = tf.reshape(image_bilinear, [1200, 1920, 3])

plt.figure()
plt.subplot(121)
plt.imshow(sess.run(image_bilinear))
plt.axis('off')
plt.subplot(122)
plt.imshow(sess.run(img))
plt.axis('off')
plt.show()

