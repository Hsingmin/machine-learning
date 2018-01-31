
# rnn_language_model.py -- Achieve a 
# natrual language processing model with deepRNN and LSTM .

import numpy as np
import tensorflow as tf
import tensorflow_ptb_reader as reader

# Constants of data
DATA_PATH = "./ptb/data"		# Dataset stored path
HIDDEN_SIZE = 200			# Hidden layer nodes 
NUM_LAYERS = 2				# deepRNN LSTM layers
VOCAB_SIZE = 10000			# Dictionary size .

# Constants of neural network
LEARNING_RATE = 1.0			# Learning rate in training process
TRAIN_BATCH_SIZE = 20			# Input data batch size
TRAIN_NUM_STEP = 35			# Training data truncate length

# Regard test data as a super long sequence for no truncating used in test process .
EVAL_BATCH_SIZE = 1			# Test data batch size
EVAL_NUM_STEP = 1			# Test data truncate length
NUM_EPOCH = 2				# Epoches of using test data
KEEP_PROB = 0.5				# Probability of no dropout for one node
MAX_GRAD_NORM = 5			# Coefficient to control gradient expansion 

# Create PTBModel to describe model and maintain state in RNN ,
# and defines ops for build neural network .
class PTBModel(object):
	def __init__(self, is_training, batch_size, num_steps):
		# Record batch size and truncate length
		self.batch_size = batch_size
		self.num_steps = num_steps

		# Define input layer with size=batch_size*num_steps ,
		# which equals to the batch size output by ptb_iterator .
		self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

		# Define expected output with size equals to real label output by ptb_iterator.
		self.targets = tf.placeholder(tf.float32, [batch_size, num_steps])

		# Set LSTM to be loop structure of deepRNN and using dropout .
		# Set state_is_tuple=True , returns (c, h) or it would be concated into a tensor .
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
		if is_training:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell, output_keep_prob=KEEP_PROB)
		
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS, state_is_tuple=True)

		# Initialize original state to zeros vector .
		self.initial_state = cell.zero_state(batch_size, tf.float32)
		# Converse word id to word vector .
		#
		# Words counts totally to VOCAB_SIZE in dictionary ,
		# word vector dimension=HIDDEN_SIZE , then variable embedding
		# dimension=VOCAB_SIZE * HIDDEN_SIZE .
		embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

		# Converse original batch_size*num_steps words' id into word vector.
		# word vector dimension=batch_size*num_steps*HIDDEN_SIZE .
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)

		# Use dropout only in training process .
		if is_training:
			inputs = tf.nn.dropout(inputs, KEEP_PROB)
		
		# Define outputs array to collect LSTM output in different moment ,
		# and get the final output through a full-connected network .
		outputs = []
		# Strore LSTM state information of different batch , and initialize to zeros .
		state = self.initial_state
		with tf.variable_scope("RNN"):
			for time_step > 0:
				tf.get_variable_scope().resuse_variables()
				# Input training data reshaped in embedding following sequence .
				cell_output, state = cell(inputs[:, time_step, :], state)

				outputs.append(cell_output)
		# Reshape output into input matrix dimension .
		output = tf.reshape(tf.concat(1, outputs), [-1, HIDDEN_SIZE])
		
		# Final full-connected layer to get the predication value ,
		# that is an array length=VOCAB_SIZE , which turned to be a
		# probability vector through softmax layer .
		weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
		bias = tf.get_variable("bias", [VOCAB_SIZE])
		logits = tf.matmul(output, weight) + bias

		# Cross Entropy loss function ,
		# Tensorflow provides sequence_loss_by_example api to calculate 
		# the cross-entropy of one sequence .
		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],				# Predication value
				[tf.reshape(self.targets, [-1])],	# Expected result 
									# reshape [batch_size, num_steps] 
									# array into one list 
				# Loss weight set to be 1 , which means the loss of 
				# different batch on different moment matters the same importance .  
				[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		# Calculate loss of every batch .
		self.cost = tf.reduce_sum(loss) / batch_size
		self.final_state = state
		
		if not is_training:
			return
		trainable_variables = tf.trainable_variables()
		# Control gradient with tf.clip_by_global_norm() to avoid gradient explosion .
		grads, _ = tf.clip_by_global_norm(
				tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

		# Define optimizer .
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

		# Define training steps .
		self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

# Batch all text contents for model to train .
def run_epoch():


# Calls run_epoch() for many times ,
# contents in text will be feeded to model for many times ,
# in which progress the arguments adjusted . 
def main(argv=None):




















































