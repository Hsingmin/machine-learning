
# regTree.py

from numpy import *

# run sunccess
def loadDataSet(fileName):

	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = list(map(float, curLine)) # map all elements to float type
		dataMat.append(fltLine)
	return dataMat

def regLeaf(dataSet):
	return mean(dataSet[:, -1])

def regErr(dataSet):
	return var(dataSet[:, -1]) * shape(dataSet)[0]

def binSplitDataSet(dataSet, feature, value):
	# mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
	# mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
	mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
	return mat0, mat1

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
	tolS = ops[0]; tolN = ops[1]
	
	if (len(set(dataSet[:, -1].T.tolist()[0])) == 1):
		print('retFeat == None, retVal = regLeaf')
		return None, leafType(dataSet)

	m, n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf; bestIndex = 0; bestValue = 0
	
	for featIndex in range(n - 1):
		# for splitVal in set(dataSet[:, featIndex]):
		for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if ((shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN)): 
				continue

			newS = errType(mat0) + errType(mat1)
			if (newS < bestS):
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if (S - bestS) < tolS:
		return None, leafType(dataSet)
	
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	
	if ((shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN)):
		return None, leafType(dataSet)

	return bestIndex, bestValue


def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None : return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	
	lSet, rSet = binSplitDataSet(dataSet, feat, val)

	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)

	return retTree		





























