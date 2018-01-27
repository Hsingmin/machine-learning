
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



















































