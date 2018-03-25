
# asyncio_demo.py

import random
import asyncio

def old_fib(n):
	res = [0]*n
	index = 0
	a = 0
	b = 1
	while index < n:
		res[index] = b
		a, b = b, a+b
		index += 1
	return res

def test_old_fib(arg = None):
	print('-'*10 + 'test old fib ' + '-'*10)
	for fib_res in old_fib(20):
		print(fib_res)

def yield_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		yield b
		a, b = b, a+b
		index += 1

def test_yield_fib(arg = None):
	print('-'*10 + 'test yield fib ' + '-'*10)
	for fib_res in yield_fib(20):
		print(fib_res)

def stupid_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_cnt = yield b
		print('let me think {0} secs'.format(sleep_cnt))
		a, b = b, a+b
		index += 1

def test_stupid_fib(arg = None):
	print('-'*10 + 'test stupid fib ' + '-'*10)
	N = 20
	sfib = stupid_fib(N)
	#fib_res = next(sfib)
	fib_res = sfib.send(None)
	while True:
		print(fib_res)
		try:
			fib_res = sfib.send(random.uniform(0, 0.5))
		except StopIteration:
			break

def copy_stupid_fib(n):
	print('Copy from stupid fib')
	yield from stupid_fib(n)
	print('Copy end')

def test_copy_stupid_fib(arg = None):
	print('-'*10 + 'copy from stupid fib ' + '-'*10)
	N = 20
	csfib = copy_stupid_fib(N)
	fib_res = next(csfib)
	while True:
		print(fib_res)
		try:
			fib_res = csfib.send(random.uniform(0, 0.5))
		except StopIteration:
			break
'''
@asyncio.coroutine
def smart_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_secs = random.uniform(0, 0.2)
		yield from asyncio.sleep(sleep_secs)
		print('Smart one think {} secs to get {}'.format(sleep_secs, b))
		a, b = b, a+b
		index += 1

@asyncio.coroutine
def dull_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_secs = random.uniform(0, 0.5)
		yield from asyncio.sleep(sleep_secs)
		print('Dull one think {} secs to get {}'.format(sleep_secs, b))
		a, b = b, a+b
		index += 1

def test_asyncio_coroutine(arg = None):
	loop = asyncio.get_event_loop()
	tasks = [asyncio.async(smart_fib(10)),
		 asyncio.async(dull_fib(10)),]
	loop.run_until_complete(asyncio.wait(tasks))
	print('All fib finished.')
	loop.close()
'''

async def smart_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_secs = random.uniform(0, 0.2)
		await asyncio.sleep(sleep_secs)
		print('Smart one think {} secs to get {}'.format(sleep_secs, b))
		a, b = b, a+b
		index += 1

async def dull_fib(n):
	index = 0
	a = 0
	b = 1
	while index < n:
		sleep_secs = random.uniform(0, 0.5)
		await asyncio.sleep(sleep_secs)
		print('Dull one think {} secs to get {}'.format(sleep_secs, b))
		a, b = b, a+b
		index += 1

def test_async_await(arg = None):
	loop = asyncio.get_event_loop()
	tasks = [asyncio.ensure_future(smart_fib(10)),
		 asyncio.ensure_future(dull_fib(10)),]
	loop.run_until_complete(asyncio.wait(tasks))
	print('All fib finished.')
	loop.close()
if __name__ == '__main__':
	test_async_await(arg = None)



































