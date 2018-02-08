
# bptt.py -- Achieve RNN BPTT with numpy .

import numpy as np

class SIMPLE_RNN:
	def __init__(self, U, V, W):
		self.u, self.v, self.w = np.asarray(U), np.asarray(V), np.asarray(W)
		self.status = None

	# Activate function .
	def activate(self, x):
		return x

	# Transform function .
	def transform(self, x):
		return x

	# Run function .
	def run(self, x):
		output = []
		# Expand inputs into array with at least 2 dimensions .  
		x = np.atleast_2d(x)
		# Initialize States to be zeros matrix from 0 moment to the final .
		self.state = np.zeros([len(x)+1, self.u.shape[0]])
		
		# Input sample in every moment .
		for t, xt in enumerate(x):
			self.state[t] = self.activate(
					self.u.dot(xt) + self.w.dot(self.state[t-1]))

			output.append(self.transform(
				self.v.dot(self.state[t])))

		return np.array(output)

#
class BPTT_RNN:


def main(argv=None):
	n_sample = 5
	# U as unit matrix 
	# V as unit matrix
	# W as double uint matrix 
	# Initialize the RNN network .
	rnn = SIMPLE_RNN(np.eye(n_sample), np.eye(n_sample), np.eye(n_sample)*2)

	# Input x an uint matrix . 
	print(rnn.run(np.eye(n_sample)))

if __name__ == '__main__':
	main()
































