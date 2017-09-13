
# treeTest.py

import trees
import treePlotter


myDat, labels = trees.createDataSet()

print('------- createDataSet --- labels : ')
print(labels)
print('------- createDataSet --- myDat : ')
print(myDat)

myTree = treePlotter.retrieveTree(0)

print('------- retrieveTree --- myTree : ')
print(myTree)

print('------- classify test data0 : ')
print(trees.classify(myTree, labels, [1, 0]))

print('------- classify test data1 : ')
print(trees.classify(myTree, labels, [1, 1]))

trees.storeTree(myTree, 'classifierStorage.txt')

print('------- classifierStorage and grab : ')
print(trees.grabTree('classifierStorage.txt'))















