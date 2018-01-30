
# ptb_data_read.py -- Read ptb raw data and converse words into number .

import tensorflow as tf 
import ptb_reader

# Ptb data stored path .
DATA_PATH = "./ptb/data"
train_data, validate_data, test_data, _ = ptb_reader.ptb_raw_data(DATA_PATH)

'''
# Read raw data .
print(len(train_data))
print(train_data[:100])
'''

# Split training data into array batch_size=4 truncated_length=5
result = ptb_reader.ptb_iterator(train_data, 4, 5)

# Read data in first batch including input in every moment and the real label .

x, y = next(result)
print("X: ", x)
print("y: ", y)









































