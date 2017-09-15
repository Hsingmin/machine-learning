
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






















