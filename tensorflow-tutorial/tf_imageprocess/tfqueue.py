
# tfqueue.py -- Queue as one kind of node in tensorflow have its own status .
#

import tensorflow as tf
import numpy as np
import threading
import time

'''
# Create a FIFO queue that can store two elements of integer type at most .
q = tf.FIFOQueue(2, "int32")

# Initialize elements in queue with enqueue_many() function .
init = q.enqueue_many(([0, 10],))

# Pop the tail element into variable x with dequeue() function .
x = q.dequeue()
y = x + 1

# Push y into queue .
q_inc = q.enqueue([y])

with tf.Session() as sess:
	# Run queue initialize operation .
	init.run()
	for _ in range(5):
		# The whole process including popping x , increment and pushing
		# will be executed when running q_inc .
		v, _ = sess.run([x, q_inc])
		print(v)
'''
'''
# Inter-threads communication with tf.Coordinator class .
# Thread quit when shoul_stop() returns True ,
# Notice the other threads quit by calling request_stop() function .

# Function running in thread .
def MyLoop(coord, worker_id):
	# Judge whethear stop and print own worker_id .
	while not coord.should_stop():
		# Stop all threads randomly .
		if np.random.rand() < 0.1:
			print("Stop from id: %d\n" % worker_id, end="")
			# Notice the other threads quit .
			coord.request_stop()
		else:
			print("Working on id : %d\n" % worker_id, end="")

		time.sleep(1)

# Create Coordination class . 
coord = tf.train.Coordinator()
# Create 5 threads .
threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]

# Start all threads .
for t in threads:
	t.start()

# Wait for all threads quiting .
coord.join(threads)
'''

# Create a fifo queue with 100 elements of real data type at most .
queue = tf.FIFOQueue(100, "float")
# Push operation to queue .
enqueue_op = queue.enqueue([tf.random_normal([1])])

# Create multiple threads to run enqueue operations .
# [enqueue_op]*5 starting 5 threads in which enqueue_op running .
qr = tf.train.QueueRunner(queue, [enqueue_op]*5)

# Add created QueueRunner into collection of Tensorflow Graph .
# In tf.train.add_queue_runner() , no collection specified , then
# add QueueRunner into tf.GraphKeys.QUEUE_RUNNERS collection defaultly .
tf.train.add_queue_runner(qr)
# Pop operation from queue .
out_tensor = queue.dequeue()

with tf.Session() as sess:
	# Coordinate started threads using tf.train.Coordinator() .  
	coord = tf.train.Coordinator()
	# Explictly calling tf.train.start_queue_runners() to start all
	# threads when QueueRunner() used , otherwise program would wait 
	# forever when calling dequeue operation .
	# 
	# tf.train.start_queue_runners() will start all QueueRunners in
	# tf.GraphKeys.QUEUE_RUNNERS collection , because it can only 
	# start QueueRunners specified in tf.train.add_queue_runner() .
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# Get value popped from queue .
	for _ in range(3):
		print(sess.run(out_tensor)[0])

	# Stop all threads with tf.train.Coordinator .
	coord.request_stop()
	coord.join(threads)
















































