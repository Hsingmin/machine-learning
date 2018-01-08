
# fabbYield.py -- demo to learn yield

def originFabb(max):
	n, a, b = 0, 0, 1
	while n < max:
		print(b)
		a, b = b, (a+b)
		n = n+1

def listFabb(max):
	n, a, b = 0, 0, 1
	L = []
	while n < max:
		L.append(b)
		a, b = b, (a+b)
		n = n+1
	return L

class iterFabb(object):

	def __init__(self, max):
		self.max = max
		self.n, self.a, self.b = 0, 0, 1

	def __iter__(self):
		return self

	def next(self):
		if self.n < self.max:
			r = self.b
			self.a, self.b = self.b, self.a+self.b
			self.n = self.n+1
			return r
		raise StopIteration()

def yieldFabb(max):
	n, a, b = 0, 0, 1
	while n < max:
		yield b
		a, b = b, a+b
		n = n+1

def readFile(fpath):
	BLOCK_SIZE = 1024
	with open(fpath, 'rb') as f:
		while True:
			block = f.read(BLOCK_SIZE)
			if block:
				yield block
			else:
				return
















