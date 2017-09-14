
# regTreesTest.py

import regTrees
from numpy import *


myDat = regTrees.loadDataSet('ex00.txt')
tMat = mat(myDat)

print('-------- loadDataSet regLeaf : ')
print(regTrees.regLeaf(tMat))

print('-------- loadDataSet regErr : ')
print(regTrees.regErr(tMat))

print('-------- tMat : ')
print(tMat)

myTree = regTrees.createTree(tMat)

print('-------- regTree createTree : ')
print(myTree)




















