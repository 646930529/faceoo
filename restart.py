import sys
import os
import time
from threading import Thread


print('runing')


def restart_program():
	python = sys.executable
	os.execl(python, python, *sys.argv)
	sys.exit(0)


def restart_auto(hour):
	t = time.time()
	lt = time.localtime(t)
	if lt.tm_hour == hour and lt.tm_min == 0:
		time.sleep(60)
		print('restart', lt)
		restart_program()


def restart_tick(hour):
	while True:
		time.sleep(30)
		restart_auto(hour)


t=Thread(target=restart_tick,args=(2,))
t.start()

