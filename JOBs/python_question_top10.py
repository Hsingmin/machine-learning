'''
class A(object):
	def show(self):
		print('base class show')

class B(A):
	def show(self):
		print('derived show')

def test0(arg=None):
	obj = B()
	# call child class object method.
	obj.show()	# derived class show
	# object __class__ method point to parent class 
	obj.__class__ = A
	obj.show()	# base class show
	# object __class__ method point back to child class
	obj.__class__ = B
	obj.show()	# derived show

class A(object):
	def __init__(self, a, b):
		self.__a = a
		self.__b = b
	def show(self):
		print('a = ', self.__a, 'b = ', self.__b)
	def __call__(self, num):
		print('call : ', self.__a + num)

def test1(arg=None):
	a = A(10, 20)
	a.show()
	a(80)
class B(object):
	def func(self):
		print('B func')
	def __init__(self):
		print('B INIT')

class A(object):
	def func(self):
		print('A func')
	def __new__(cls, a):
		print('NEW', a)
		if a > 10:
			return super(A, cls).__new__(cls)
		return B()
	def __init__(self, a):
		print('INIT', a)

def test2(arg=None):
	a1 = A(5)	# NEW 5 , B INIT
	a1.func()	# B func
	a2 = A(20)	# NEW 20 , INIT 20
	a2.func()	# A func
def test3(arg=None):
	ls = [1,2,3,4]
	list1 = [i for i in ls if i>2]
	print(list1)	# [3,4]

	list2 = [i*2 for i in ls if i>2]
	print(list2)	# [6,8]

	dict1 = {x: x**2 for x in (2,4,6)}
	print(dict1)	# {2: 4, 4: 16, 6: 36}

	dict2 = {x: 'item' + str(x**2) for x in (2,4,6)}
	print(dict2)	# {2: 'item4', 4: 'item16', 6: 'item36'}

	set1 = {x for x in 'hello world' if x not in 'low level'}
	print(set1)	# set[('h', 'r', 'd')]

num = 9
def func1():
	global num
	num = 20

def func2():
	print(num)

def test4(arg=None):
	func2()		# 9
	func1()
	func2()		# 20
'''

def test5(arg=None):
	a = 8
	b = 9
	print('before swap : a = ', a, 'b = ', b)	# 8 9
	a, b = b, a
	print('after swap : a = ', a, 'b = ', b)	# 9 8

class A(object):
	def __init__(self, a, b):
		self.__a = a
		self.__b = b
		print('INIT')
	
	def default_method(self, *args):
		print('default method : ' + str(args[0]))

	def __getattr__(self, name):
		print('undefined method : ', name)
		return self.default_method

def test6(arg=None):
	a1 = A(10, 20)
	a1.func1(33)
	a1.func2('hello')
	a1.func3(10)

def closure(num):
	def inner(value):
		return num * value
	return inner

def test7(arg=None):
	z = closure(7)
	print(z(9))

def test8(num):
	str = 'first'
	for i in range(num):
		str += "x"
	return str

if __name__ == '__main__':
	print(test8(10))































