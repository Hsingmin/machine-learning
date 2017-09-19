
# kNNTest.py

import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

group, labels = kNN.createDataSet()
print('group : ', group)
print('labels : ', labels)

classResult = kNN.classify0([0, 0], group, labels, 3)

# print(classResult)
'''
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels)
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()

normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

print('normMat : ', normMat)
print('ranges : ', ranges)
print('minVals : ', minVals)
'''

# kNN.datingClassTest()

# kNN.classifyPerson()

testVector = kNN.img2vector('digits/testDigits/0_13.txt')
print(testVector[0, 0 : 31])
print(testVector[0, 32 : 63])






















