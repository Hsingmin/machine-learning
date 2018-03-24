
# -*- coding: 	utf-8		-*-
# -*- version:	python 3.4.5	-*-
import time

def foo():
	print('in foo()')

# Define time clock as wrapper
def timeit(func):
	
	# Define inline wrapper
	def wrapper():
		start = time.clock()
		func()
		end = time.clock()
		print('used: ', end - start)
		
	return wrapper

foo = timeit(foo)

if __name__ == '__main__':
	foo()





















