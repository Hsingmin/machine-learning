
# fpGrowthTest.py

import fpGrowth
from numpy import *

# FP-Tree node create test

rootNode = fpGrowth.treeNode('pyramid', 9, None)

rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)

rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)

rootNode.disp()
































