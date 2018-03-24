
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

if __name__ == '__main__':
	test3()





























