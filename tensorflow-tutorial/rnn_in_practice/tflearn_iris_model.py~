
# tflearn_iris_model.py -- Creatd model with high-level package TFLearn
# integreted in tf.contrib.learn .
# Use scikit-learn model to preprocess dataset .

import tensorflow as tf
from sklearn import  cross_validation
from sklearn import datasets
from sklearn import metrics

# Import tf.contrib.learn module and tf.contrib.layers model 
# to construct full-connected network model .
learn = tf.contrib.learn
layers = tf.contrib.layers

# Define model , return the predicted value , loss and train_op 
# on input features and target .
def custom_model(features, target):
	# Converse terget into one-hot incode format ,
	# category 1 as [1,0,0]
	# category 2 as [0,1,0]
	# category 3 as [0,0,1]
	target = tf.one_hot(target, 3, 1, 0)

	# models.logistic_regression() has been removed from tensorflow.contrib.learn 
	# Get a singel layer full-connected neural network with high-level API in TFLearn .
	# logits, loss = learn.models.logistic_regression(features, target)
	logits = tf.contrib.layers.fully_connected(features, 3)	
	loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

	# Create model optimizer , and get optimize step and train_op .
	train_op = tf.contrib.layers.optimize_loss(
			loss,						# Loss function
			tf.contrib.framework.get_global_step(),		# Train steps and update when training
			optimizer='Adagrad',				# Optimizer
			learning_rate=0.01)				# Learning rate
	
	# Return predication on input dataset , loss and train_op .
	return tf.arg_max(logits, 1), loss, train_op

# Load iris dataset and split into train dataset and test dataset with 
# cross-validation method .
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
		test_size=0.2, random_state=0)

# Package custom model with tf.contrib.learn.Estimator()
classifier = learn.Estimator(model_fn=custom_model)

# Run 100 steps to train custom model .
classifier.fit(x_train, y_train, steps=10000)

# Predict the result with trained model .
y_predicted = classifier.predict(x_test)

# Calculate accuracy of the model .
# score = metrics.accuracy_score(y_test, y_predicted)
# print('Accuracy: %.2f%%' % (score * 100))
y_ = [y for y in y_predicted]
print('y_predicted :	' , y_)
print("predict sequence length = ", len(y_))
print("=====================================================")
print('y_test :	' , y_test)
print("test dataset label length = ", len(y_test))

score = metrics.accuracy_score(y_test, y_)
print('Accuracy: %.2f%%' % (score * 100))








































