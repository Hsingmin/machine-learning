
# name_variable_scope.py -- make difference of tensorflow.name_scope()
# and tensorflow.variable_scope() .
# 
import tensorflow as tf
'''
with tf.name_scope("hello") as name_scope:
	arr1 = tf.get_variable("arr1", shape=[2,10], dtype=tf.float32)
	
	print('++++++++++++ tf.name_scope() practice ++++++++++++')
	# Output: hello/ 
	# tf.name_scope() method returns a string 
	print(name_scope)

	# Output: arr1:0
	# Variable defined in name_scope have no prefix 'hello/'
	print(arr1.name)

	# Output: "scope_name :"
	# tf.get_variable_scope().original_name_scope() returns null 
	print("scope_name : %s" %(tf.get_variable_scope().original_name_scope))
'''
with tf.variable_scope("hello") as variable_scope:
	arr1 = tf.get_variable("arr1", shape=[2,10], dtype=tf.float32)

	print('++++++++++++ tf.variable_scope() practice ++++++++++++')

	# Output: <tensorflow.python.ops.variable_scope.VariableScope object at 0x7fbc09959210>
	# tf.variable_scope() returns an object
	print(variable_scope)
	
	# Output: hello
	# variable_scope.name attribute 
	print(variable_scope.name)

	# Output: hello/arr1:0
	# Variable name defined in variable_scope with prefix 'hello/'
	print(arr1.name)

	# Output: hello/
	# variable_scope as variable prefix 
	print(tf.get_variable_scope().original_name_scope)

	# Output: hello/arr2/
	# Nested path with prefix 'hello/arr2/'
	with tf.variable_scope('arr2') as v_scope2:
		print(tf.get_variable_scope().original_name_scope)

print('++++++++++++ tf.variable_scope() and tf.name_scope() mixture practice ++++++++++++')
with tf.name_scope("name_scope1"):
	with tf.variable_scope("variable_scope1"):
		w = tf.get_variable("w", shape=[2])
		res = tf.add(w,[3])
# Output: variable_scope1/w:0
# 
print(w.name)
# Output: name_scope1/variable_scope1/Add:0
#
print(res.name)

# name_scope will add prefix to operation but variable created by
# get_variable() not .
# 
# tf.get_variable_scope() returns variable_scope , despite name_scope
#
# We can ignore name_scope when using tf.get_variable_scope().reuse_variables()

with tf.name_scope("name_scope2") as nscope2:
	with tf.name_scope("name_scope3") as nscope3:
		print(nscope3)

with tf.variable_scope("variable_scope2") as vscope3:
	with tf.variable_scope("variable_scope3") as vscope3:
		print(vscope3.name)






















