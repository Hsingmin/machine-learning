
# kmeansTest.py

import kMeans

from numpy import *

# dataMat = mat(kMeans.loadDataSet('testSet.txt'))

'''
print('min(dataMat[:, 0]) = ', min(dataMat[:, 0]))

print('min(dataMat[:, 1]) = ', min(dataMat[:, 1]))

print('max(dataMat[:, 0]) = ', max(dataMat[:, 0]))

print('max(dataMat[:, 1]) = ', max(dataMat[:, 1]))

print('randCent of dataset : ', kMeans.randCent(dataMat, 2))
print('distance of eclud : ', kMeans.distEclud(dataMat[0], dataMat[1]))
'''

# myCentroids, clusterAssing = kMeans.kMeans(dataMat, 4)

# print('myCentroids : ', myCentroids)
# print('clusterAssing : ', clusterAssing)

dataMat3 = mat(kMeans.loadDataSet('testSet2.txt'))

centList, newAssments = kMeans.biKmeans(dataMat3, 3)

print('centList = ', centList)














































































