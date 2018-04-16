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

import codecs
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Flatten

visible = Input(shape=(128, 322, 3))
m = TimeDistributed(Flatten(), name='timedistrib')(visible)
output = Dense(10, activation='sigmoid')(m)
model = Model(inputs=visible, outputs=output)
# print(model.summary())
with codecs.open('./model.txt', 'w', 'utf-8') as file:
    print(str(model.summary()), file=file)

