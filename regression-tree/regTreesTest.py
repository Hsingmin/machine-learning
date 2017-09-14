
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

myDat2 = regTrees.loadDataSet('ex2.txt')
myMat2 = mat(myDat2)
myTree2 = regTrees.createTree(myMat2, ops = (0, 1))
print('-------- regTree createTree2 : ')
print(myTree2)

myDat3 = regTrees.loadDataSet('ex2test.txt')
myMat2Test = mat(myDat3)

regTrees.prune(myTree2, myMat2Test)

myMat4 = mat(regTrees.loadDataSet('exp2.txt'))
print(regTrees.createTree(myMat4, regTrees.modelLeaf, regTrees.modelErr, (1, 10)))





















