
# svdTest.py

from numpy import *

import svdRec

'''
U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])

print('matrix U : ')
print(U)
print('vector Sigma : ')
print(Sigma)
print('matrix VT : ')
print(VT)
'''

Data = svdRec.loadExData()
U, Sigma, VT = linalg.svd(Data)

print('triagular matrix Sigma : ')
print(Sigma)

Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
Data3 = U[:, :3] * Sig3 * VT[:3, :]

print('SVD Data3 : ')
print(Data3)

















