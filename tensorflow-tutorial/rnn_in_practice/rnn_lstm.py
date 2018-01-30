
# rnn_lstm.py -- Forward propagation of RNN with LSTM structure .
# Variables used in LSTM will defined in 
# tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

import tensorflow as tf

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# Initialize state in LSTM to be zeros , 
# just like the other network , LSTM use one batch of training data 
# as input , BasicLSTMCell class provides function zero_state to produce 
# zeros initial state .
state = lstm.zero_state(batch_size, tf.float32)

# Loss function .
loss = 0.0

# Limit the input sequence length to avoid gradient vanishing problem .
#
# num_steps as the maximum length of input sequence here .
for i in range(num_steps):
	# Declare variables of LSTM in the first time ,
	# reuse the defined variables in following time .
	if i > 0: 
		tf.get_variable_scope().reuse_variables()

	# Deal with one moment in time series in every step .
	# 
	# Pass arguments current input current_input, former time state,
	# to get the current LSTM output lstm_ouput and updated current 
	# time state .
	lstm_output, state = lstm(current_input, state)
	
	# Pass current LSTM output into the final full-connected network .
	final_output = fully_connected(lstm_ouput)
	# Calculate loss in current time .
	loss += calc_loss(final_output, exepected_output)


# Deep RNN structure .
# A simple LSTM as deepRNN loop basic structure .
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

# Get dropout with tf.nn.rnn_cell.DropoutWrapped() .
#
# argument :
#	input_keep_prob --  input nodes dropout probability .
#	output_keep_prob -- output nodes dropout probability .
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)

# Get deepRNN forward propagation in every moment in time series , 
# number_of_layers as how many layers in one moment from x_t to h_t .
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)

# Get initiali state with zero_state() function . 
state = stacked_lstm.zero_state(batch_size, tf.float32)

# Calculate forward propagation value of every moment in time series .
for i in range(num_steps):
	if i > 0:
		tf.get_variable_scope().reuse_variables()
		stacked_lstm_output, state = stacked_lstm(current_input, state)
		final_output = fully_connected(stacked_lstm_output)
		loss += calc_loss(final_output, expected_output)










































