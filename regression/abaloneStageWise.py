
# abaloneStageWise.py

import regression
from numpy import *
import matplotlib.pyplot as plt

xArray, yArray = regression.loadDataSet('abalone.txt')

xArray = array(xArray, dtype = float)
yArray = array(yArray, dtype = float)

# regression.stageWise(xArray, yArray, 0.01, 200)

stageWeights = regression.stageWise(xArray, yArray, 0.001, 5000)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(stageWeights)
plt.show()

print(' ---- lwlr output for this dataset : ---- ')

xMat = mat(xArray)
yMat = mat(yArray).T
xMat = regression.regularize(xMat)
yM = mean(yMat, 0)
yMat = yMat - yM
weights = regression.standRegres(xMat, yMat.T)
print('weights.T : ', weights.T)






























