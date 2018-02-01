
# iterator_yeild.py 

class Fabb(object):
	def __init__(self, max):
		self.max = max
		self.n, self.a, self.b = 0, 0, 1
	
	def __iter__(self):
		return self

	def __next__(self):
		if self.n < self.max:
			r = self.b
			self.a, self.b = self.b, self.a + self.b
			self.n = self.n + 1
			return r

		raise StopIteration()

# Using yield instead of class .
def y_fabb(max):
	n, a, b = 0, 0, 1
	while n < max:
		yield b
		a, b = b, a + b
		n += 1

def main(argv=None):
	
	print('Using class Fabb generate : ')
	for n in Fabb(10):
		print(n)

	print("===============================")
	print("Using yield keyword : ")
	for n in y_fabb(10):
		print (n)

if __name__ == '__main__':
	main()



































