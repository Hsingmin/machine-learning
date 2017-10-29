
# svdRec.py

from numpy import *
from numpy import linalg as la

def euclidSim(inA, inB):
	return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
	if len(inA) < 3:
		return 1.0
	return 0.5 + 0.5 / corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
	num = float(inA.T * inB)
	denom = la.norm(inA) * la.norm(inB)
	return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):






























