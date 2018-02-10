
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

# RNN network with BPTT .
class BPTT_RNN(SIMPLE_RNN):
	# Sigmoid activate function .
	def activate(self, x):
		return 1 / (1 + np.exp(-1))

	# Softmax converse function .
	def transform(self, x):
		safe_exp = np.exp(x - np.max(x))
		return safe_exp / np.sum(safe_exp)

	def bptt(self, x, y):
		x, y, n = np.asarray(x), np.asarray(y), len(y)

		# Get outputs and calculate State .
		o = self.run(x)
		dis = o - y
		# dv = (o - y) * state .
		dv = dis.T.dot(self.state[:-1])
		
		# Initialize du and dw to zeros vector .
		du = np.zeros_like(self.u)
		dw = np.zeros_like(self.w)

		# From moment n-1 to -1
		for t in range(n-1, -1, -1):
			st = self.state[t]
			ds = self.v.T.dot(dis[t]) * st * (1 - st)
			#
			for bptt_step in range(t, max(-1, t-10), -1):
				du += np.outer(ds, x[bptt_step])
				dw += np.outer(ds, self.state[bptt_step])
				st = self.state[bptt_step-1]
				ds = self.w.T.dot(ds) * st * (1-st)

		return du, dv, dw

	def loss(self, x, y):
		o = self.run(x)
		return np.sum(
				-y * np.log(np.maximum(o, 1e-12))
				- (1-y) * np.log(np.maximum(1-o, 1e-12)))


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
































