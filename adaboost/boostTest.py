
# boostTest.py

from numpy import *
import boost

def loadSimpleData():
	dataMat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])

	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return dataMat, classLabels

dataMat, classLabels = loadSimpleData()

D = mat(ones((5, 1)) / 5)

print(boost.buildStump(dataMat, classLabels, D))

classifierArray = boost.adaBoostTrainDS(dataMat, classLabels, 30)
print('classifierArray = ', classifierArray)

# boost.adaClassify([0, 0], classifierArray)
# boost.adaClassify([[5, 5], [0, 0]], classifierArray)

# real sample : horseColic problem
dataArray, labelArray = boost.loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = boost.adaBoostTrainDS(dataArray, labelArray, 10)

testArray, testLabelArray = boost.loadDataSet('horseColicTraining2.txt')
prediction10 = boost.adaClassify(testArray, classifierArray)

errorArray = mat(ones((len(testLabelArray), 1)))
print(errorArray[prediction10 != mat(testLabelArray).T].sum())

boost.plotROC(aggClassEst.T, labelArray)












