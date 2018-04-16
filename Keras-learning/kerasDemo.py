
# kerasDemo.py

from keras import *

# import os
# os.environ["THEANO_FLAGS"]="device=gpu0"

model = Sequential()

# hidden layer1
model.add(Dense(input_dim=28*28, output_dim=500))
model.add(Activation('sigmoid'))
model.add(dropout(0.8))

# hidden layer2
model.add(Dense(output_dim=500))
# model.add(Activation('sigmoid'))
model.add(Activation('relu'))
model.dropout(dropout(0.8))

# softmax layer
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

model.compile(loss='mse',\
	      optimizer=SGD(lr=0.1),\
	      metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',\
#	      optimizer=SGD(lr=0.1),
#	      metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',\
		optimizer=Adam(),\
		metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, nb_epoch=20)

score = model.evaluate(x_test, y_test)
print('Total loss on Testing Set: ', score[0])
print('Accuracy of Testing Set: ', score[1])

result = model.predict(x_test)
















