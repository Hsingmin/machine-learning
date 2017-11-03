
# svdTest.py

from numpy import *
from numpy import linalg as la

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

'''
Data = svdRec.loadExData()
U, Sigma, VT = linalg.svd(Data)

print('triagular matrix Sigma : ')
print(Sigma)

Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
Data3 = U[:, :3] * Sig3 * VT[:3, :]

print('SVD Data3 : ')
print(Data3)
'''

'''
simMat = mat(svdRec.loadExData())
print('euclid similarity = ')
print(svdRec.euclidSim(simMat[:, 0], simMat[:, 4]))

print('euclid similarity = ')
print(svdRec.euclidSim(simMat[:, 0], simMat[:, 0]))

print('cosin similarity = ')
print(svdRec.cosSim(simMat[:, 0], simMat[:, 4]))

print('cosin similarity = ')
print(svdRec.cosSim(simMat[:, 0], simMat[:, 0]))

print('pearson similarity = ')
print(svdRec.pearsSim(simMat[:, 0], simMat[:, 4]))

print('pearson similarity = ')
print(svdRec.pearsSim(simMat[:, 0], simMat[:, 0]))
'''

'''
recommendMat = mat(svdRec.loadExData())
recommendMat[0,1] = recommendMat[0,0] = recommendMat[1,0] = recommendMat[2,0] = 4
recommendMat[3,3] = 2
print('recommendMat : ')
print(recommendMat)

print('recommend user2 with cosine Similarity : ')
print(svdRec.recommend(recommendMat, 2))

print('recommend user2 with euclid Similarity : ')
print(svdRec.recommend(recommendMat, 2, simMeas = svdRec.euclidSim))

print('recommend user2 with pearson Similarity : ')
print(svdRec.recommend(recommendMat, 2, simMeas = svdRec.pearsSim))
'''

'''
U, Sigma, VT = la.svd(mat(svdRec.loadExData2()))
print('Sigma of ExData2 : ')
print(Sigma)
Sig2 = Sigma ** 2
print('total power of ExData2 : ')
print(sum(Sig2))
print('90% of total power : ')
print(sum(Sig2) * 0.9)
print('estimated power from Sigma 1 to 3 : ')
print(sum(Sig2[: 3]))
'''
'''
svdMat = mat(svdRec.loadExData2())
svdRec.recommend(svdMat, 1, estMethod = svdRec.svdEst, simMeas = svdRec.pearsSim)
'''

print('SVD compress image : ')
svdRec.imgCompress(2)


