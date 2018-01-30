
# rnn_loop.py

import numpy as np

# Input sequence .
X = [1, 2]
# Former state input .
state = [0.0, 0.0]

# Split weight matrix for different input part .
# State weight matrix .
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
# Input weight matrix .
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# Output layer weight .
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# Run forward propagation chronologically .
for i in range(len(X)):
	# Calculate full-conneted network in loop .
	before_activation = np.dot(state, w_cell_state) + \
			X[i] * w_cell_input + b_cell
	state = np.tanh(before_activation)

	# Calculate final output .
	final_output = np.dot(state, w_output) + b_output

	#
	print("before activation: " , before_activation)
	print("state: ", state)
	print("final_output: ", final_output)








































