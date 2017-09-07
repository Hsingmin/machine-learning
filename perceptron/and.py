
from pcp import Perceptron

def f(x):
	'''
	activator
	'''
	return 1 if x > 0 else 0

def get_training_dataset():

	'''
	get training dataset based on TRUE-FALSE table
	'''
	input_vecs = [[1,1], [0,0], [1,0], [0,1]]

	# expected output
	labels = [1, 0, 0, 0]

	return input_vecs, labels

def train_and_perceptron():
	
	'''
	train data based on TRUE-FALSE table
	'''

	# create a perceptron
	p = Perceptron(2, f)

	# train and iterate 100 rounds ita = 0.1
	input_vecs, labels = get_training_dataset()
	
	# print(input_vecs)
	# print(labels)

	p.train(input_vecs, labels, 10, 0.1)

	return p

if __name__ == '__main__':

	# train and-perceptron
	and_perception = train_and_perceptron()

	print(and_perception)

	print('1 and 1 = %d' % and_perception.predict([1,1]))

	print('1 and 0 = %d' % and_perception.predict([1,0]))

	print('0 and 1 = %d' % and_perception.predict([0,1]))
	
	print('0 and 0 = %d' % and_perception.predict([0,0]))

