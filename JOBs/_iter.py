
# -*- coding:	utf-8		-*-
# -*- version:	python 3.5.4	-*-
# _iter.py

import sys

i = iter(range(10000))

print("id(i.__next__()) = ", id(i.__next__()))

print("sys.getsizeof(i) = ", sys.getsizeof(i))

print("sys.getsizeof(i.__next__()) = ", sys.getsizeof(i.__next__()))

e = range(10000)
print("sys.getsizeof(e) = ", sys.getsizeof(e))
print("sys.getsizeof(list(e))", sys.getsizeof(list(e)))



































