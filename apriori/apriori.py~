
# apriori.py

from numpy import *

def loadDataSet():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])

	C1.sort()
	return map(frozenset, C1)

def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				#if not ssCnt.has_key(can):
				if not can in ssCnt:
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(list(D)))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support
	return retList, supportData

# create Ck
def aprioriGen(Lk, k):
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i + 1, lenLk):
			L1 = list(Lk[i])[: k - 2]; L2 = list(Lk[j])[: k - 2]
			L1.sort(); L2.sort()
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])
	return retList

def aproiri(dataSet, minSupport = 0.5):
	C1 = createC1(dataSet)
	D = map(set, dataSet)
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while(len(L[k - 2]) > 0):
		Ck = aprioriGen(L[k - 2], k)
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK)
		L.append(Lk)
		k += 1
	return L, supportData

def generateRules(L, supportData, minConf = 0.7):
	bigRuleList = []
	for i in range(1, len(L)):
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if(i > 1):
				rulesFromConseq(freqSet, H1, \
						supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList

def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet] / supportData[freqSet - conseq]
		if conf >= minConf:
			print(freqSet - conseq, '-->', conseq, 'conf: ', conf)
			br1.append((freqSet - conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
	m = len(H[0])
	if(len(freqSet) > (m + 1)):
		Hmp1 = aprioriGen(H, m + 1)
		Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
		if(len(Hmp1) > 1):
			rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)
'''
from time import sleep
from votesmart import votesmart
votesmart.apikey = '49024thereoncewasamanfromnantucket94040'
def getActionIds():
	actionIdList = []; billTitleList = []
	fr = open('recent20bills.txt')
	for line in fr.readlines():
		billNum = int(line.split('\t')[0])
		try:
			billDetail = votesmart.votes.getBill(billNum)
			for action in billDetail.actions:
				if action.level == 'House' and \
					(action.stage == 'Passage' or \
					action.stage == 'Amendment Vote'):
					actionId = int(action.actionId)
					print('bill : %d has actionId : %d' %(billNum, actionId))
					actionIdList.append(actionId)
					billTitleList.append(line.strip().split('\t')[1])
		except:
			print('problem getting bill %d' %billNum)
		sleep(1)
	return actionIdList, billTitleList

def getTransList(actionIdList, billTitleList):
	itemMeaning = ['Republican', 'Democratic']
	for billTitle in billTitleList:
		itemMeaning.append('%s -- Nay' % billTitle)
		itemMeaning.append('%s -- Yea' % billTitle)
	transDict = {}
	voteCount = 2
	for actionId in actionIdList:
		sleep(3)
		print('getting votes for actionId : %d' %actionId)
		try:
			voteList = votesmart.votes.getBillActionVotes(actionId)
			for vote in voteList:
				if not transDict.has_key(vote.candidateName):
					transDict[vote.candidateName] = []
					if vote.officeParties == 'Democratic':
						transDict[vote.candidateName].append(1)
					elif vote.officeParties == 'Republican':
						transDict[vote.candidateName].append(0)
				if vote.action == 'Nay':
					transDict[vote.candidateName].append(voteCount)
				elif vote.action == 'Yea':
					transDict[vote.candidateName].append(voteCount + 1)

		except:
			print('problem getting actionId: %d' %actionId)
		voteCount += 2
	return transDict, itemMeaning

'''







































