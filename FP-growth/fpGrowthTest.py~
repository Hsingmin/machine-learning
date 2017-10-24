
# fpGrowthTest.py

import fpGrowth
from numpy import *

'''
# FP-Tree node create test

rootNode = fpGrowth.treeNode('pyramid', 9, None)

rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)

rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)

rootNode.disp()
'''

simData = fpGrowth.loadSimpleData()
print('simData : ' , simData)

initSet = fpGrowth.createInitSet(simData)
print('initSet : ', initSet)

simFPTree, simHeaderTable = fpGrowth.createTree(initSet, 3)
simFPTree.disp()

'''
print('========= prefix path : ')
print(fpGrowth.findPrefixPath('x', simHeaderTable['x'][1]))
print(fpGrowth.findPrefixPath('z', simHeaderTable['z'][1]))
print(fpGrowth.findPrefixPath('r', simHeaderTable['r'][1]))
'''

freqItems = []
fpGrowth.mineTree(simFPTree, simHeaderTable, 3, set([]), freqItems)































