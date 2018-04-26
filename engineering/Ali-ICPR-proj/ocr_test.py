#coding:utf-8
# ocr_test.py
import ocr.model as om
import numpy as np
from PIL import Image
import ocr.keys as keys
import os
from keras.utils import plot_model

test_path = './tmp/test'
h5_path = 'd:/python_work/h5/model3.4.h5'

def get_test_samples(path):
    samples = []
    for rootdir, subdirs, filenames in os.walk(path):
        for file in filenames:
            samples.append(os.path.join(path, file))
    return samples

if __name__ =='__main__':
    characters = keys.alphabet[:]
    height = 32
    nclass = len(characters)
    model, basemodel = om.get_model(height, nclass)
    pred_model = om.load_model(basemodel, h5_path)
    test_samples = get_test_samples(test_path)
    for sample in test_samples:
        try:
            im = Image.open(sample)
            result = om.predict(im, pred_model)
            print("---------------------------------------")
            print(result)
        except Exception as e:
            print(e)
            pass









