
# ccnetwork.py -- Collection Computated Network with L2-Regularizer Loss-Function .

import tensorflow as tf

# Get weights on the edge of one layer, and the L2-Regularizer Loss to collection named 'losses' .
#
# lambda has been a keyword , and may conflict in python3.x , programmers must avoid using it 
# to name a variable .
def get_weight(shape, lamda):
	# Produce a Variable .
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

	# Add L2-Regularizer loss function into collection named 'losses' through tensorflow API .
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamda)(var))

	# Return produced variable var .
	return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8

# Define node numbers of every layers .
#
# 2-inputs hidden_layer1 hidden_layer2 hiddend_layer3 and output_layer .
layer_dimension = [2, 10, 10, 10, 1]
# Layers of Neural Network = 5 .
n_layers = len(layer_dimension)

# Set a variable to save the node of current deepest layer ,
# current layer initial value is input layer x .
cur_layer = x
# Number of nodes in current layer ,
# weights connect between two layers defined as 'in_dimension' and 'out_dimension' .
# in_dimension intialized as layer_dimension[0] also known as input layer x .
in_dimension = layer_dimension[0]

# Produce a 5-layer Full-Connected Neural Network .
for i in range(1, n_layers):
	# layer_dimension[i] as next layer's node number .
	out_dimension = layer_dimension[i]
	# Produce current layer's weights variable , and save them into collection 'losses' .
	#
	# Weights regularizer loss function between in_dimension and out_dimension is regarded 
	# as a part of total loss-function with a regularization coefficient = 0.001 .
	weight = get_weight([in_dimension, out_dimension], 0.001)
	bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

	# Forward Propagation overlay using ReLU activated function .
	#
	# cur_layer variable keeps the current layer's output .
	cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight)+bias)
	# Update next layer's nodes and get into next iteration .
	in_dimension = layer_dimension[i]

# All L2-Regularizer Loss-Function have been already added into collection 'losses' when 
# setting the forward propagation .
#
# Calculate the model loss-function on training dataset .
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# Add MSE into collection 'losses' .
tf.add_to_collection('losses', mse_loss)

# get_collection() returns a list including all elements that would be added into 
# the loss function .
loss = tf.add_n(tf.get_collection('losses'))


# ExponentialMovingAverage .
# Define a variable to computate the moving average with initial value 0 .
v1 = tf.Variable(0, dtype=tf.float32)
# Use step to simulate iterations in neural network training , and also control
# decay rate .
step = tf.Variable(0, trainable=False)

# Define a Moving-Average class with initial value decay_rate = 0.99 and variable step .
ema = tf.train.ExponentialMovingAverage(0.99, step)
# Define an operation to update the moving average and a list that will be 
# refreshed every time operation executed .
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
	# Initialize all variables .
	init_ops = tf.global_variables_initializer()
	sess.run(init_ops)

	# Get Moving-Average value of v1 .
	print(sess.run([v1, ema.average(v1)]))

	# Update v1 value to 5 .
	sess.run(tf.assign(v1, 5))
	# Get the Moving-Average value of v1 with decay rate=min(0.99, (1+step)/(10+step)=0.1)=0.1
	# shadow_variable = shadow_variable*decay_rate + (1-decay_rate)*variable
	# Output = 0*0.1 + 5*(1-0.1) = 4.5
	#
	# tensorflow.train.ExponentialMovingAverage(0.99, step) function keeps a list to save 
	# calculation value refreshed every time .
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)]))

	# Set step value = 10000 .
	sess.run(tf.assign(step, 10000))
	# Set v1 value = 10 .
	sess.run(tf.assign(v1, 10))
	
	# Update v1 moving-average value with decay rate = min(0.99, (1+step)/(10+step)=0.999)=0.99 .
	# v1 moving-average value will updated to be 4.5*0.99 + 10*(1-0.99) = 4.555 .
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)]))
	
	# Update v1 moving-average value again .
	sess.run(maintain_averages_op)
	print(sess.run([v1, ema.average(v1)]))


































































