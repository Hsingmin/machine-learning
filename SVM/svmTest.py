
# svmTest.py

from numpy import *

import svmMLiA as sl

dataArray, labelArray = sl.loadDataSet('testSet.txt')

# print(dataArray)

# print(labelArray)

'''
b, alphas = sl.smoSimple(dataArray, labelArray, 0.6, 0.001, 40)

# print('b = ', b)
print('alphas numbers = ', shape(alphas[alphas > 0]))

for i in range(100):
	if alphas[i] > 0.0:
		print(dataArray[i], labelArray[i])

'''

b, alphas = sl.smoP(dataArray, labelArray, 0.6, 0.001, 40)

ws = sl.calcWs(alphas, dataArray, labelArray)
print('ws : ', ws)

dataMat = mat(dataArray)
print('first data label : ', dataMat[0] * mat(ws) + b)
print('labelArray[0] = ', labelArray[0])

print('labelArray[1] = ', labelArray[0])

print('labelArray[2] = ', labelArray[0])




























