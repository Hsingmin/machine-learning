#coding:utf-8
# ocr_test.py
import ocr.model as om
import numpy as np
from PIL import Image
import ocr.keys as keys
import os
from keras.utils import plot_model
import train.train as tt

test_path = './tmp/test'
h5_path = 'd:/python_work/h5/model4.01.h5'

def get_test_samples(path):
    samples = []
    for rootdir, subdirs, filenames in os.walk(path):
        for file in filenames:
            samples.append(os.path.join(path, file))
    return samples
"""
# Batch provider for ocr model training.
def batch_loader(category=None):
    if category == None:
        return
    dataset = load_dataset(category, INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    batch_list = []
    label_list = []
    X_batch = []
    y_batch = []

    for i in range(BATCH_SIZE):
        img_dir = dataset[np.random.randint(len(dataset))]
        batch_list.append(img_dir)
        label_list.append(os.path.basename(img_dir).split('.')[0])

    for img_dir in batch_list:
        image_raw = Image.open(img_dir)
        # image = np.array(image_raw.convert('RGB'))
        alligned_height = 32
        bimage = image_raw.convert('L')
        scale = bimage.size[1]*1.0/alligned_height
        width = int(bimage.size[0]/scale)
        image = bimage.resize((width, alligned_height))
        # image.save(os.path.join('./to', os.path.basename(img_dir)))
        image = np.array(image)
        # print(image)
        X_batch.append(image)

    aligned_batch = dd.AlignedBatch(alligned_height, TRUNCATED_WIDTH)
    X_batch = np.array(aligned_batch(X_batch))
    # print(label_list)
    aligned_onehot = dd.AlignedOnehot(N_LEN, characters)
    y_batch = np.array(aligned_onehot(label_list))

    return X_batch, y_batch

def input_allocate(X_batch, y_batch):
    X_batch = X_batch.reshape((BATCH_SIZE, 32, -1, 1))
    batch_size = X_batch.shape[0]
    input_length = int(X_batch.shape[2]/4)-2
    label_length = y_batch.shape[1]
    X, Y = [X_batch, y_batch, np.ones(batch_size)*input_length,
            np.ones(batch_size)*label_length], np.ones(batch_size)
    return X, Y
"""

if __name__ =='__main__':
    characters = keys.alphabet[:]
    height = 32
    nclass = len(characters)
    model, basemodel = om.get_model(height, nclass)
    pred_model = om.load_model(basemodel, h5_path)
    # test_samples = get_test_samples(test_path)
    X_test, y_test = tt.batch_loader(category='test')
    samples, _ = tt.input_allocate(X_test, y_test)

    for sample in samples:
        try:
            # im = Image.open(sample)
            result = pred_model.predict(sample)
            print("---------------------------------------")
            print(result)
        except Exception as e:
            print(e)
            pass









