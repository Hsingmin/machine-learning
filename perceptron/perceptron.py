
# perceptron.py

import random

def sign(v):
    if v > 0:
        return 1
    else:
        return -1


def training():
    train_data1 = [[1, 3, 1], [2, 5, 1], [3, 8, 1], [2, 6, 1]] 
    train_data2 = [[3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]
    train_datas = train_data1 + train_data2

    weight = [0, 0]  
    bias = 0  
    learning_rate = 0.5  

    train_num = int(input("train num: "))  

    for i in range(train_num):
        train = random.choice(train_datas)
        x1, x2, y = train
        predict = sign(weight[0] * x1 + weight[1] * x2 + bias)  
        print("train data: x: (%d, %d) y: %d  ==> predict: %d" % (x1, x2, y, predict))
        if y * predict <= 0:  
            weight[0] = weight[0] + learning_rate * y * x1  
            weight[1] = weight[1] + learning_rate * y * x2
            bias = bias + learning_rate * y  
            print("update weight and bias: "),
            print(weight[0], weight[1], bias)

    print("stop training: "),
    print(weight[0], weight[1], bias)

    return weight, bias


# test
def test():
    weight, bias = training()
    while True:
        test_data = []
        data = input('enter test data (x1, x2): ')
        if data == 'q': break
        test_data += [int(n) for n in data.split(',')]
        predict = sign(weight[0] * test_data[0] + weight[1] * test_data[1] + bias)
        print("predict ==> %d" % predict)



















