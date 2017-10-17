
# aprioriTest.py

import apriori

from numpy import *

dataSet = apriori.loadDataSet()

print('aprioriDataSet = ', dataSet)

C1 = apriori.createC1(dataSet)
print('apriori C1 = ', C1)

D = map(set, dataSet)
print('apriori D = ', D)

L1, supportData0 = apriori.scanD(D, C1, 0.5)
print('apriori L1 = ', L1)
print('apriori supportData0 = ', supportData0)






















