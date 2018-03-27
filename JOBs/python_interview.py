
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
	def __init__(self, data=None, next=None):
		self.val = data
		self.next = next
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
		mid_pivot = list[0]
		smaller_list = [e for e in list[1:] if e <= mid_pivot]
		bigger_list = [e for e in list[1:] if e > mid_pivot]
		final_list = quick_sort(smaller_list) + [mid_pivot]\
			+ quick_sort(bigger_list)
		return final_list

def test_quick_sort(arg=None):
	list = [2,4,6,7,1,2,5]
	print(quick_sort(list))

# BiTree
class BTNode(object):
	def __init__(self, data, left=None, right=None):
		self.data = data
		self.left = left
		self.right = right

def bitree_level_traverse(root):
	if not root:
		return
	row = [root]
	while row:
		print([node.data for node in row])
		row = [kid for item in row for kid in (item.left, item.right) if kid]

def bitree_pre_traverse(root):
	if not root:
		return
	print(root.data)
	bitree_pre_traverse(root.left)
	bitree_pre_traverse(root.right)

def bitree_mid_traverse(root):
	if root == None:
		return
	if root.left is not None:
		bitree_mid_traverse(root.left)
	print(root.data)
	if root.right is not None:
		bitree_mid_traverse(root.right)

def bitree_post_traverse(root):
	if root == None:
		return
	if root.left is not None:
		bitree_post_traverse(root.left)
	if root.right is not None:
		bitree_post_traverse(root.right)
	print(root.data)

def bitree_get_deepth(root):
	if not root:
		return 0
	return max(bitree_get_deepth(root.left), bitree_get_deepth(root.right))+1

def bitree_is_same(p, q):
	if None == p and None == q:
		return True
	elif p and q:
		return p.data == q.data and bitree_is_same(p.left, q.left) \
				and bitree_is_same(p.right, q.right)
	else:
		return False

def bitree_rebuild(pre_order, mid_order):
	if not pre_order:
		return
	current_node = pre_order[0]	# root
	index = mid_order.index(pre_order[0])
	current_node.left = rebuild(pre_order[1:index+1], mid_order[:index])
	current_node.right = rebuild(pre_order[index+1:], mid_order[index+1:])
	return current_node

def test_bitree(arg=None):
	tree = BTNode(1, BTNode(3, BTNode(7, BTNode(0)), BTNode(6)), BTNode(2, BTNode(5), BTNode(4)))
	print('Level Traverse BiTree : ')
	bitree_level_traverse(tree)
	print('Pre-Order Traverse BiTree : ')
	bitree_pre_traverse(tree)
	print('Mid-Order Traverse BiTree : ')
	bitree_mid_traverse(tree)
	print('Post-Order Traverse BiTree : ')
	bitree_post_traverse(tree)
	print('Deepth of tree is : %d' %bitree_get_deepth(tree))

	another_tree = BTNode(1, BTNode(3, BTNode(7, BTNode(1)), BTNode(6)), BTNode(2, BTNode(5), BTNode(4)))
	print('tree and another_tree is the same : ', bitree_is_same(tree, another_tree))

def reverse_linklist(link):
	pre = link
	cur = link.next
	pre.next = None
	while cur:
		tmp = cur.next
		cur.next = pre
		pre = cur
		cur = tmp
	return pre

def test_reverse_linklist(arg=None):
	link = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6, ListNode(7, ListNode(8, ListNode(9)))))))))
	root = reverse_linklist(link)
	while root:
		print(root.val)
		root = root.next

def check_change_word(s1, s2):
	alist1 = list(s1)
	alist2 = list(s2)

	alist1.sort()
	alist2.sort()

	pos = 0
	matches = True

	while pos < len(s1) and matches:
		if alist1[pos] == alist2[pos]:
			pos += 1
		else:
			matches = False
	return matches

def test_check_change_word(arg=None):
	s1 = 'list'
	s2 = 'still'
	print('list and still is change word : ', check_change_word(s1, s2))

	s1 = 'stop'
	s2 = 'post'
	print('stop and post is change word : ', check_change_word(s1, s2))

# Level Order Traverse a Directory
import os

def traverse_directory(path):
	sub_path_list = []
	file_path_list = []
	for rootdir, subdirs, filenames in os.walk(path):
		for subdir in subdirs:
			sub_path_list.append(os.path.join(rootdir, subdir))

		for filename in filenames:
			file_path_list.append(os.path.join(rootdir, filename))
	return sub_path_list, file_path_list

def test_traverse_directory(arg=None):
	path = './sub_pack'
	sub_directories, files = traverse_directory(path)
	print('sub directory list is : ', sub_directories)
	print('file list is : ', files)

def get_median(list):
	list = sorted(list)
	length = len(list)
	return list[int(length/2)]

def test_get_median(arg=None):
	list = [1,5,6,3,6,9,5]
	m = get_median(list)
	print(m)

if __name__ == '__main__':
	test_get_median(arg=None)





























