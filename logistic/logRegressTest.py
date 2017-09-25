
# logRegressTest.py

import logRegres

from numpy import *

dataArray, labelMat = logRegres.loadDataSet()

# print('dataArray : ', dataArray)
# print('labelArray : ', labelMat)

# weights = (logRegres.gradAscent(dataArray, labelMat))
# logRegres.plotBestFit(weights.getA())

# weights = (logRegres.stockGradAscent0(array(dataArray), labelMat))
#logRegres.plotBestFit(weights)

# weights = (logRegres.stockGradAscent1(array(dataArray), labelMat))
# logRegres.plotBestFit(weights)

logRegres.multiTest()























