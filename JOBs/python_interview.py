
import time
import threading
from multiprocessing import Process

def timer(no, interval):
	cnt = 0
	while cnt < 10:
		print('Thread: (%d) Time: %s\n' %(no, time.ctime()))
		time.sleep(interval)
		cnt += 1

def test0(arg = None):
	t1 = threading.Thread(target=timer, args=(1, 1))
	t2 = threading.Thread(target=timer, args=(2, 2))
	t1.start()
	t2.start()
	t1.join()
	t2.join()

class NewThread(threading.Thread):
	
	def set_timer(self, no, interval):
		self.no = no
		self.interval = interval
	def run(self):
		timer(self.no, self.interval)
	
def test1(arg = None):
	t1 = NewThread()
	t2 = NewThread()
	t1.set_timer(1,1)
	t2.set_timer(2,2)
	t1.start()
	t2.start()
	t1.join()
	t2.join()

def ptimer(no, interval):
	cnt = 0
	while cnt < 10:
		print('Process: (%d) Time: %s\n' %(no, time.ctime()))
		time.sleep(interval)
		cnt += 1

def test2(arg = None):
	p1 = Process(target=ptimer, args=(1,1))
	p2 = Process(target=ptimer, args=(2,2))
	
	p1.start()
	p2.start()
	p1.join()
	p2.join()

import asyncio

async def cor1():
	print("COR1 start ")
	await cor2()
	print("COR1 end ")

async def cor2():
	print("COR2 ")

def test3():
	loop = asyncio.get_event_loop()
	loop.run_until_complete(cor1())
	loop.close()

def get_value(l, r, c):
	return l[r][c]

def find_element(l, x):
	m = len(l) - 1
	n = len(l[0]) - 1
	r = 0
	c = n	# point to the last column 
	while c >= 0 and r <= m:
		value = get_value(l, r, c)
		if value == x:
			return True
		elif value > x:
			c -= 1
		elif value < x:
			r += 1
	return False

def test_find_element(arg = None):
	l = [[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]]
	x1 = 5
	x2 = 13
	x3 = 0
	print(find_element(l, x1), find_element(l, x2), find_element(l, x3))

class ListNode(object):
	def __init__(self, x):
		self.val = x
		self.next = None
class LinkList(object):
	def __init__(self):
		self.head = 0

	def initializer(self, data):
		self.head = ListNode(data[0])
		p = self.head
		for v in data[1:]:
			p.next = ListNode(v)
			p = p.next

def swap_pairs(head):
	if head != None and head.next != None:
		next = head.next
		head.next = swap_pairs(next.next)
		next.next = head
		return next
	return head
	
def test_swap_pairs(arg = None):
	data = [1,2,3,4]
	ls = LinkList()
	ls.initializer(data)
	ls_head = ls.head
	swap_ls = []
	node = swap_pairs(ls_head)
	for i in range(len(data)):
		swap_ls.append(node.val)
		node = node.next
	print(swap_ls)

def loop_merge_sort(l1, l2):
	tmp = []
	while len(l1) > 0 and len(l2) > 0:
		if l1[0] < l2[0]:
			tmp.append(l1[0])
			del(l1[0])
		else:
			tmp.append(l2[0])
			del(l2[0])
	tmp.extend(l1)
	tmp.extend(l2)
	return tmp

def test_loop_merge_sort(arg = None):
	a = [1,2,3,7]
	b = [4,5,6]
	print(loop_merge_sort(a, b))

def cross_node(l1, l2):
	length1 = 0
	length2 = 0
	p1 = l1.head
	p2 = l2.head
	while p1:
		p1 = p1.next
		length1 += 1
	p1 = l1.head
	print('length1 = %d' %length1)
	while p2:
		p2 = p2.next
		length2 += 1
	p2 = l2.head
	print('length2 = %d' %length2)
	if length1 > length2:
		for _ in range(length1 - length2):
			p1 = p1.next
	else:
		for _ in range(length2 - length1):
			p2 = p2.next
	while p1 != None:
		if(p1.val != p2.val):
			p1 = p1.next
			print(p1.val, end='')
			p2 = p2.next
			print(' ', p2.val)
		else:
			return p1
	print('No Cross Node')
	return None

def test_cross_node(arg=None):
	data1 = [1,2,3,7,9,1,5]
	data2 = [4,5,7,9,1,5]
	l1 = LinkList()
	l1.initializer(data1)
	l2 = LinkList()
	l2.initializer(data2)

	xnode = cross_node(l1, l2)
	if xnode != None:
		print(xnode.val)
	else:
		print('No cross node of l1 and l2')

def binary_search(list, element):
	start = 0
	end = len(list)-1
	while start <= end:
		middle = int((start+end)/2)
		guess = list[middle]
		if guess == element:
			return middle
		elif guess > element:
			end = middle-1
		else:
			start = middle+1
	return None

def test_binary_search(arg=None):
	list = [1,3,4,5,7,8,9]
	element = 3
	print('get element 3 index is %d' 
		%(binary_search(list, element)))

def quick_sort(list):
	if len(list)<2:
		return list
	else:


if __name__ == '__main__':
	test_binary_search(arg = None)





























