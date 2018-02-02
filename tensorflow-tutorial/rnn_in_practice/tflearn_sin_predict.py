
# tflearn_sin_predict.py -- Create model to predict sin() function with TFLearn .

import numpy as np
import tensorflow as tf

#
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow.contrib.learn as learn


HIDDEN_SIZE = 30			# LSTM size 
NUM_LAYERS = 2				# LSTM layers
TIMESTEPS = 10				# Truncate length
TRAIN_STEPS = 10000			# Train epoches
BATCH_SIZE = 32				# Batch size of training data

TRAINING_EXAMPLES = 10000		# Training dataset size
TESTING_EXAMPLES = 1000			# Testing dataset size
SAMPLE_GAP = 0.01			# Sample gap

def generate_data(seq):
	X = []
	y = []
	
	# X array like [[1,2,3], [2,3,4], [3,4,5], ...]
	# y array like [[4], [5], [6], ...]
	for i in range(len(seq) - TIMESTEPS - 1):
		X.append([seq[i: i + TIMESTEPS]])
		y.append([seq[i + TIMESTEPS]])

	return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Create LSTM model 
def lstm_model(X, y):
	# Multiple layers LSTM .
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
	cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
	# In high version of Tensorflow , tf.pack() and tf.unpack() have been
	# removed , use tf.stack() and tf.unstack() instead .
	x_ = tf.unstack(X, axis=1)

	# tf.nn.rnn() has been removed in high version of Tensorflow ,
	# use tf.nn.dynamic_rnn() instead .
	output, _ = tf.nn.dynamic_rnn(cell, x_, dtype=tf.float32)

	# Focus on the last element in array output .
	output = output[-1]

	# No module named models.linear_regression ,
	# Create one full-connected layer to predict the output .
	prediction = layers.fully_connected(output, len(y))
	# 
	loss = tf.contrib.losses.mean_suared_error(prediction, y)

	# Creatd model optimizer and train_op .
	train_op = tf.contrib.layers.optimize_loss(
			loss, tf,contrib.framework.get_global_step(),
			optimizer="Adagrad", learning_rate=0.1)

	return prediction, loss, train_op

# Create deepRNN model .
regressor = learn.Estimator(model.fn=lstm_model)

# Produce training dataset and test dataset with sin() function .
#
# numpy.linspace(start, end, length) returns an arithmetic sequence .
test_start = TRAINING_EXAMPLES * SAMPLE_GAP		
test_end = (TRAINING_EXAMPLES + TRAINING_EXAMPLES) * SAMPLE_GAP

train_X, train_y = generate_data(np.sin(np.linspace(
	0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
	test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# Train model with fit() .
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAIN_STEPS)

# Get prediction of test data input into model .
predicted = [[pred] for pred in regressor.predict(test_X)]
# Mean square error .
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is : %f " % rmse[0])

fig = plt.figure()
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.saving('sin_predicted.png')



































