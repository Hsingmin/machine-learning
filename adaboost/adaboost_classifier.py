
# adaboost_classifier.py

def G1(x):
	if x < 2.5:
		return 1
	else:
		return -1

def G2(x):
	if x < 8.5:
		return 1
	else:
		return -1

def G3(x):
	if x > 5.5:
		return 1
	else:
		return -1

def G(x):
	a1 = .4236
	a2 = .6496
	a3 = .7514

	if a1 * G1(x) + a2 * G2(x) + a3 *  G3(x) >0:
		return 1
	elif a1 * G1(x) + a2 * G(x) + a3 * G3(x) <0:
		return -1
	else:
		return 0
	
a1 = .4236
a2 = .6496
a3 = .7514
for i in range(10):
	print(i)
	print('corresponding classify result : ')

	f = a1 * G1(i) + a2 * G2(i) + a3 * G3(i)
	if f > 0:
		f = 1
	elif f < 0:
		f = -1
	else:
		f = 0
	print(f)
