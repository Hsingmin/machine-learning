
# Perceptron 

from functools import reduce

class Perceptron(object):
	def __init__(self, input_num, activator):
		'''
		initialize perceptron, set no. of params and activator
		activator dtype : double -> double
		'''
		
		self.activator = activator
		
		# set weights vecs(input_num * 1) to 0
		self.weights = [0.0 for _ in range(input_num)]

		# set bias to 0
		self.bias = 0.0

	def __str__(self):
		'''
		print out the weights and bias
		'''
		return 'weights\t : %s\nbias\t : %f\n' % (self.weights, self.bias)

	def predict(self, input_vec):
		'''
		input vecs, output predict values
		'''
		# zip input_vec[x1, x2, x3, ...] and weights[w1, w2, w3, ...]
		# turned into [(x1, w1), (x2, w2), (x3, w3), ...]
		# map function get [x1*w1, x2*w2, x3*w3, ...]
		# reduce function to get sum

		# print('UPDATED WEIGHTS : ')
		# for each in self.weights:
		# 	print(each)
		# print('UPDATED BIAS : ')
		# print(self.bias)
		# print('+++++++++++++++++++++++++++++++++')

		return self.activator(\
				reduce(lambda a, b : a + b, \
				map(lambda x_w : x_w[0] * x_w[1], \
				zip(input_vec, self.weights)), \
				0.0) + self.bias)

	def train(self, input_vecs, labels, iteration, rate):
		'''
		input train data : a group of vecs and corresponding label
		training rounds and ita
		'''
		for i in range(iteration):
			self._one_iteration(input_vecs, labels, rate)

	def _one_iteration(self, input_vecs, labels, rate):
		'''
		one iteration
		'''

		# zip input data and output data into [(input_vec, label), ...]
		# training sample as (input_vec, label)
		samples = zip(input_vecs, labels)
		
		# for each in samples:
			# print(each)
		# update weights to each sample
		for (input_vec, label) in samples :
			
			# get output of perceptron of current weights
			print(input_vec)
			print(label)
			output = self.predict(input_vec)
			print('_one_iteration_ : output = ')
			print(output)
			# update weights
			self._update_weights(input_vec, output, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		'''
		update weights
		'''

		# zip input_vec[x1, x2, x3, ...] and weights[w1, w2, w3, ...]
		# turned into [(x1, w1), (x2, w2), (x3, w3), ...]
		# update weights
		delta = label - output

		self.weights = map(lambda x_w : x_w[1] + rate * delta * x_w[0], \
				zip(input_vec, self.weights))

		print('weights and bias : ')
		for each in self.weights:
			print(each)
		# update bias
		self.bias += rate * delta
		print(self.bias)








