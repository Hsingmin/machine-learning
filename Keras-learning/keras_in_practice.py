# -*- coding: utf-8 -*-
#
# -*- version:
#		keras 2.1.2
#		tensorflow 1.4.0 
#		python 3.5.2
# -*- author: Hsingmin Lee
#
# keras_in_practice.py -- Keras framework with Tensorflow as backend,
#
# Construct model with different methods providen by Keras.

"""
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

# Create independent input layer,
# shape is a tuple representing input tensor dimension.
#
# Perceptron with 10-features input.
visible = Input(shape=(10,), name='visible')
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)

# Summarize layers
print(model.summary())
# Plot graph
plot_model(model, to_file='multilayer_perceptron_graph.png')
"""
"""
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

visible = Input(shape=(64, 64, 1))                              #image width=64, height=64, channel=1
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)   #kernel size=4*4*32
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                   #maxpooling, size=2*2
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)     #kernel size=4*4*16
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                   #maxpooling, size=2*2
hidden1 = Dense(10, activation='relu')(pool2)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)

# Summarize layers
print(model.summary())
# Plot graph
plot_model(model, to_file='cnn_graph.png')
"""
"""
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM

visible = Input(shape=(100, 1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)

# Summarize layers
print(model.summary())
# Plot graph
plot_model(model, to_file='lstm_graph.png')
"""
"""
# Shared input layer model.
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

visible = Input(shape=(64, 64, 1))

conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)

conv2 = Conv2D(16, kernel_size=8, activation='relu')(visible)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)

merge = concatenate([flat1, flat2])

hidden1 = Dense(10, activation='relu')(merge)

output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)

# Summarize layers
print(model.summary())
# Plot graph
plot_model(model, to_file='shared_input_layer_graph.png')
"""
"""
# Shared feature extraction layer.
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate

visible = Input(shape=(100, 1))
# Feature extraction.
extract1 = LSTM(10)(visible)
# First interpretation model.
interp1 = Dense(10, activation='relu')(extract1)
# Second interpretation model.
interp21 = Dense(10, activation='relu')(extract1)
interp22 = Dense(20, activation='relu')(interp21)
interp23 = Dense(10, activation='relu')(interp22)

merge = concatenate([interp1, interp23])

output = Dense(1, activation='sigmoid')(merge)
model = Model(inputs=visible, outputs=output)

# Summarize layers
print(model.summary())
# Plot graph
plot_model(model, to_file='shared_feature_extractor_graph.png')
"""
"""
import codecs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten
from keras.layers import GRU, Bidirectional

visible = Input(shape=(128, 322, 3))
m = TimeDistributed(Flatten(), name='timedistrib')(visible)
# m = Bidirectional(GRU(256, return_sequences=True), name='bgru')(m)
m = GRU(256, return_sequences=True, name='gru')(m)
output = Dense(10, activation='sigmoid')(m)
model = Model(inputs=visible, outputs=output)
print(model.summary())
plot_model(model, to_file='bgru_model_graph.png')
"""

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import SGD
import numpy as np
import random

def lambda_func(args):
    y_pred, label = args
    return abs(y_pred - label)

def get_model():
    visible = Input(shape=(10,), name='visible')
    hidden1 = Dense(10, activation='relu')(visible)
    hidden2 = Dense(20, activation='relu')(hidden1)
    hidden3 = Dense(10, activation='relu')(hidden2)
    y_pred = Dense(1, activation='sigmoid')(hidden3)
    basemodel = Model(inputs=visible, outputs=y_pred)

    label = Input(shape=[None,], name='label')
    loss_out = Lambda(lambda_func, output_shape=(1,), name='percep')([y_pred, label])
    model = Model(inputs=[visible, label], outputs=[loss_out])
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True, clipnorm=5)

    basemodel.compile(loss={'percep': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    return model, basemodel

def show_model():
    model, basemodel = get_model()
    # Summarize layers
    print(model.summary())
    # Plot graph
    plot_model(model, to_file='multilayer_perceptron_graph.png')

def gen():
    X_train = np.random.random(size=10)
    y_train = np.array(random.randint(0,9))
    yield (X_train, y_train)

def train_op(data, labels):
    model, basemodel = get_model()
    # model.fit_generator(gen(), steps_per_epoch=100, epochs=2000)
    basemodel.fit(data, labels, epochs=10, batch_size=100)
    print('train model.')

if __name__ == '__main__':
    X_train = np.random.random((1000, 10))
    y_train = np.random.randint(2, size=(1000, 1))
    train_op(X_train, y_train)

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
"""












