
# regression.py

from numpy import *

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().strip().split('\t')) - 1
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArray = []
		currLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArray.append(currLine[i])
		dataMat.append(lineArray)
		labelMat.append(float(currLine[-1]))
	
	return dataMat, labelMat

def standRegres(xArray, yArray):
	xMat = mat(xArray); yMat = mat(yArray).T
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0.0:
		print('This matrix is singular, cannot do inverse')
		return 

	ws = xTx.I * (xMat.T * yMat)

	return ws

def lwlr(testPoint, xArray, yArray, k = 1.0):
	xMat = mat(xArray); yMat = mat(yArray).T
	m = shape(xMat)[0]
	weights = mat(eye((m)))

	for j in range(m):
		diffMat = testPoint - xMat[j, :]
		weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))

	xTx = xMat.T * (weights * xMat)
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular, cannot do inverse")
		return

	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws

def lwlrTest(testArray, xArray, yArray, k = 1.0):
	m = shape(testArray)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArray[i], xArray, yArray, k)
	
	return yHat

def rssError(yArray, yHatArray):
	return ((yArray - yHatArray) ** 2).sum()

def ridgeRegres(xMat, yMat, lam = 0.2):
	xTx = xMat.T * xMat
	denom = xTx + eye(shape(xMat)[1]) * lam
	if linalg.det(denom) == 0.0:
		print('This Matrix is singular, cannot do inverse')
		return

	ws = denom.I * (xMat.T * xMat)
	return ws

def ridgeTest(xArray, yArray):
	xMat = mat(xArray); yMat = mat(yArray).T
	yMean = mean(yMat, 0)
	yMat = yMat - yMean
	xMeans = mean(xMat, 0)
	xVar = var(xMat, 0)
	xMat = (xMat - xMeans) / xVar
	numTestPts = 30

	wMat = zeros((numTestPts, shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat, yMat, exp(i - 10))
		wMat[i, :] = ws.T

	return wMat

def stageWise(xArray, yArray, eps = 0.01, numInt = 100):
	xMat = mat(xArray); yMat = mat(yArray).T
	yMean = mean(yMat, 0)
	yMat = yMat - yMean
	xMat = regularize(xMat)
	m, n = shape(xMat)
	returnMat = zeros((numInt, n))
	ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
	for i in range(numInt):
		print(ws.T)
		lowestError = inf;
		for j in range(n):
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign
				yTest = xMat * wsTest
				rssE = rssError(yMat.A, yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i, :] = ws.T
	return returnMat


	
	














































