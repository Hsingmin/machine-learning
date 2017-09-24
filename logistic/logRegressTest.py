
# logRegressTest.py

import logRegres

dataArray, labelMat = logRegres.loadDataSet()

# print('dataArray : ', dataArray)
# print('labelArray : ', labelMat)

weights = (logRegres.gradAscent(dataArray, labelMat))

logRegres.plotBestFit(weights.getA())



























