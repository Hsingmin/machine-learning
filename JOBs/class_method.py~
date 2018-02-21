
# -*- 	coding: utf-8	 	-*-
# -*- 	versison: python 3.4.5 	-*-

# class_method.py

class Test(object):
	num_of_instance = 0
	name_of_instance = "Object1"
	queue_of_instance = []
	
	def __init__(self, name):
		self.name = name
		Test.num_of_instance += 1
		Test.name_of_instance = name
		Test.queue_of_instance.append(name)

if __name__ == '__main__':
	print("Class attribute : ")
	print("Test.num_of_instance = ", Test.num_of_instance)		# 0
	print("Test.name_of_instance = ", Test.name_of_instance)	# "Object"
	print("Test.queue_of_instance = ", Test.queue_of_instance)	# []

	# Create object named 'jack'
	t1 = Test('jack')
	print("After instantiation : ")
	print("Test.num_of_instance = ", Test.num_of_instance)		# 1
	print("Test.name_of_instance = ", Test.name_of_instance)	# "jack"
	print("Test.queue_of_instance = ", Test.queue_of_instance)	# ['jack']
	
	# Object t1 shares class variable 
	print("t1.num_of_instance = ", t1.num_of_instance)		# 1
	print("t1.name_of_instance = ", t1.name_of_instance)		# "jack"
	print("t1.queue_of_instance = ", t1.queue_of_instance)		# ['jack']

	# Create object named 'tom'
	t2 = Test('tom')
	print("Test.num_of_instance = ", Test.num_of_instance)		# 2
	print("Test.name_of_instance = ", Test.name_of_instance)	# "tom"
	print("Test.queue_of_instance = ", Test.queue_of_instance)	# ['jack', 'tom']
	
	# Object t2 shares class variable
	print("t2.num_of_instance = ", t2.num_of_instance)		# 2
	print("t2.name_of_instance = ", t2.name_of_instance)		# "tom"
	print("t2.queue_of_instance = ", t2.queue_of_instance)		# ['jack', 'tom']
	# number, string as immutable variable
	# list as mutable variable
	t1.name_of_instance = "jack"
	print("t1.num_of_instance = ", t1.num_of_instance)		# 2
	print("t1.name_of_instance = ", t1.name_of_instance)		# "jack"
	print("t1.queue_of_instance = ", t1.queue_of_instance)		# ['jack', 'tom']






























