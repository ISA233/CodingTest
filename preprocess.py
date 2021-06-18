import numpy as np
import gzip
import os
import torch
from torch import nn
from config import config
import torch.nn.functional as F

pool = nn.MaxPool2d(4)


def pool2(a):
	b = np.zeros((a.shape[0], a.shape[1] // 4, a.shape[2] // 4), dtype=np.uint8)
	step = 100000
	for i in range(0, b.shape[0], step):
		j = i + step
		with torch.no_grad():
			b[i: j] = pool(torch.tensor(a[i: j], dtype=torch.float32, device=config.device)).cpu().numpy().astype(np.uint8)
	return b


def main():
	for i in range(1, 6):
		path = config.path + '%d/replay_logs/' % i
		files = os.listdir(path)
		for file in files:
			if '$store$_observation_ckpt' not in file:
				continue
			print(file)
			with gzip.open(path + file, 'rb') as fi:
				a = np.load(fi)
				a = np.pad(a, ((0, 0), (6, 6), (6, 6)))
				a = pool2(a)
				with gzip.open(path + file[:19] + '_small' + file[19:], 'wb', compresslevel=3) as fo:
					fo.write(a)


if __name__ == '__main__':
	main()
