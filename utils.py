import matplotlib.image
import time
import numpy as np
import os
import re
import torch.nn.functional as F


def time_str():
	return time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime())


def save_img_pair(a, b, info):
	a = (a * 255)
	b = (b * 255)
	a[0, 0] = 255
	a[0, 1] = 0
	b[0, 0] = 255
	b[0, 1] = 0
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	a = F.interpolate(a.view(1, 1, *a.shape), scale_factor=8)
	b = F.interpolate(b.view(1, 1, *b.shape), scale_factor=8)
	a = a.view(*a.shape[2:]).numpy().astype(np.uint8)
	b = b.view(*b.shape[2:]).numpy().astype(np.uint8)
	matplotlib.image.imsave('output/%02d_a.jpeg' % info, a, cmap='gray')
	matplotlib.image.imsave('output/%02d_b.jpeg' % info, b, cmap='gray')


def check(files, new_file):
	ptn = re.compile('^' + new_file)
	for file in files:
		if ptn.search(file) is not None:
			return False
	return True


class Logger:
	def __init__(self):
		self.file = None
	
	def set_file(self):
		files = os.listdir('log/')
		cnt = 0
		while True:
			cnt += 1
			file = 'log_%d' % cnt
			if check(files, file):
				self.file = 'log/%s.txt' % file
				break
		self.write_line('Time: %s' % time_str())
	
	def write(self, text):
		assert self.file is not None
		print('log %s: %s' % (self.file, text))
		logfile = open(self.file, 'a')
		logfile.write(text)
		logfile.close()
	
	def write_line(self, text):
		self.write(text + '\n')
	
	def __call__(self, text):
		self.write_line(text)


log = Logger()
