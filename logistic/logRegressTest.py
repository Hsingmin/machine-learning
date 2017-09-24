
# logRegressTest.py

import logRegres

dataArray, labelMat = logRegres.loadDataSet()

print('dataArray : ', dataArray)
print('labelArray : ', labelArray)

print(logRegres.gradAscent(dataArray, labelMat))





























