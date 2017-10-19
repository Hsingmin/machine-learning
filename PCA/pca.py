
# pca.py

from numpy import *

def loadDataSet(fileName, delim = '\t'):
	fr = open(fileName)
	stringArray = [line.strip().split(delim) for line in fr.readlines()]
	dataArray = [map(float, line) for line in stringArray]

	return mat(dataArray)

def pca(dataMat, topNfeat = 9999999):
	meanVals = mean(dataMat, axis = 0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar = 0)
	eigVals, eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[: - (topNfeat + 1) : -1]
	redEigVects = eigVects[:, eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat

def replaceNanWithMean():
	dataMat = loadDataSet('secom.data', ' ')
	numFeat = shape(dataMat)[1]
	for i in range(numFeat):
		meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])

		dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal

	return dataMat











































