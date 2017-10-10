
# regressionTest.py

import regression
from numpy import *

import matplotlib.pyplot as plt

xArray, yArray = regression.loadDataSet('ex0.txt')
xArray = array(xArray, dtype = float)
yArray = array(yArray, dtype = float)
# print(xArray[0 : 2])
# print(yArray[0])
# print(regression.lwlr(xArray[0], xArray, yArray, 1.0))


ws = regression.standRegres(xArray, yArray)

# print(ws)

xMat = mat(xArray)
yMat = mat(yArray)
yHat = xMat * ws

# print(corrcoef(yHat.T, yMat))

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()
'''

# LWLR with Gaussion Kernel
# k = 0.01 to get a perfect regression value
yHat = regression.lwlrTest(xArray, xArray, yArray, 0.01)

# print(yHat)

sortInd = xMat[:, 1].argsort(0)
xSort = xMat[sortInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[sortInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArray).T.flatten().A[0], s = 2, c = 'red')
plt.show()
















