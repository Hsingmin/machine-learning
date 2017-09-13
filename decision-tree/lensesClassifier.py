
# lensesClassfier.py

import trees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

lensesTree = trees.createTree(lenses, lensesLabels)

print('lensesTree : ')
print(lensesTree)

treePlotter.createPlot(lensesTree)
