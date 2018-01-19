
import tensorflow as tf
import random
def inference(x):
	with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
		weights = tf.get_variable("weights", shape=[5, 1],\
		initializer=tf.truncated_normal_initializer(stddev=0.1))
		layer = tf.matmul(x, weights)
	return layer

def train():
	y = inference(tf.random_normal([1, 5], stddev=1.0))
	with tf.Session() as sess: 
		for i in range(100):
			sess.run(y)	
			print(y.eval())

def main(argv=None):
	train()

if __name__ == '__main__':
	tf.app.run()























