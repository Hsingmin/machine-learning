
# ptb_data_read.py -- Read ptb raw data and converse words into number .

import tensorflow as tf 
import tensorflow_ptb_reader as reader

# Ptb data stored path .
DATA_PATH = "./ptb/data"
train_data, validate_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

'''
# Read raw data .
print(len(train_data))
print(train_data[:100])
'''

# Split training data into array batch_size=4 truncated_length=5
result = reader.ptb_iterator(train_data,
		4,
		5)

# Read data in first batch including input in every moment and the real label .

# generator has no attribute 'next' .
# x, y = result.next()
#
# Python built-in function next() returns next value of iterable object .
x, y = next(result)
print("X: ", x)
print("y: ", y)









































