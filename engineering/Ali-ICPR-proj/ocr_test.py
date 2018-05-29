#coding:utf-8
# ocr_test.py
import ocr.model as om
import numpy as np
from PIL import Image
import ocr.keys as keys
import os
from keras.utils import plot_model
import train.train as tt
import data.dataset as dd
import codecs

test_path = 'd:/python_work/framedemo/Ali-ICPR-proj/tmp/test'
# h5_path = 'd:/python_work/h5/model19.30.h5'
h5_path = 'd:/python_work/h5/model487.92.h5'

TRUNCATED_WIDTH = 256
N_LEN = 10

def get_test_samples(path):
    samples = []
    for rootdir, subdirs, filenames in os.walk(path):
        for file in filenames:
            samples.append(os.path.join(path, file))
    return samples

# Sample provider for ocr model testing.
def sample_loader(img_dir):
    X_sample = []
    y_label = []
    image_raw = Image.open(img_dir)
    alligned_height = 64
    bimage = image_raw.convert('L')
    image = np.array(bimage)
    X_sample.append(image)
    y_label.append(os.path.basename(img_dir).split('.')[0])

    X_sample = np.array(X_sample)
    aligned_onehot = dd.AlignedOnehot(N_LEN, characters)
    y_label = np.array(aligned_onehot(y_label))

    X = X_sample.reshape((1, 64, 256, 1))
    y = y_label
    return X, y

if __name__ =='__main__':
    characters = keys.alphabet[:]
    height = 64
    nclass = len(characters)
    model, basemodel = om.get_model(height, nclass)
    pred_model = om.load_model(basemodel, h5_path)
    test_samples = get_test_samples(test_path)

    for sample in test_samples:
        # Preprocess image into array for 'the_input'
        X, y = sample_loader(sample)
        result = om.predict(X, pred_model)
        # result = pred_model.predict(X)
        print("---------------------------------------")
        print(result)
        # t = result.argmax(axis=2)[0]
        # print(t)








