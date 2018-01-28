
# example_producer.py -- Produce sample data for tf.train.string_input_producer() demo .

import tensorflow as tf

# Create TFRecord file helpler function .
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Simulate writing massive data into different TFRecord file .
#
# num_shards : how many files to write .
# instances_per_shard : how many data to write into every file .

num_shards = 2
instances_per_shard = 2

for i in range(num_shards):
	# Produce files in name with suffix of 0000n-of-0000m ,
	# 0000n as current file number ,
	# 0000m as total files number .
	filename = ('./data/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
	writer = tf.python_io.TFRecordWriter(filename)

	# Package data into Example structure and writing into TFRecord file .
	for j in range(instances_per_shard):
		#
		example = tf.train.Example(features=tf.train.Features(feature={\
				'i': _int64_feature(i),
				'j': _int64_feature(j)}))
		writer.write(example.SerializeToString())
	writer.close()





































