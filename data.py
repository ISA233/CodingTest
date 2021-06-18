import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import gzip
import torch
from torch import nn
from typing import List
from config import config

from torch.utils.data import Dataset, DataLoader


class GameLoader(Dataset):
	def __init__(self, subdir, block_id, mode='small', lr=None):
		assert mode == 'origin' or mode == 'small'
		self.mode = mode
		self.block_id = block_id
		self.subdir = subdir
		self.lr = None
		if mode == 'origin':
			self.lr = (0, 10000)
		
		self.buffer_length: int
		self.observation_buffer: np.ndarray
		self.action_buffer: np.ndarray
		self.reward_buffer: np.ndarray
		self.terminal_buffer: np.ndarray
		self.terminal_idx: List
		
		self.length = 0
		self.idx_dict = dict()
		
		self._load_buffer()
		self._preprocess()
	
	def _load_buffer(self):
		print('GameLoader:_load_buffer (%d, %d)' % (self.subdir, self.block_id))
		path = config.path + '%d/replay_logs/' % self.subdir
		# with gzip.open(path + '$store$_observation_small_ckpt.%d.gz' % self.block_id, 'rb') as f:
		if self.mode == 'small':
			with gzip.open(path + '$store$_observation_small_ckpt.%d.gz' % self.block_id, 'rb') as f:
				self.observation_buffer = np.frombuffer(f.read(), dtype=np.uint8)
		else:
			with gzip.open(path + '$store$_observation_ckpt.%d.gz' % self.block_id, 'rb') as f:
				self.observation_buffer = np.load(f)
		self.observation_buffer = self.observation_buffer.reshape((1000000, config.image_size, config.image_size))
		with gzip.open(path + '$store$_action_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.action_buffer = np.load(f)
		with gzip.open(path + '$store$_reward_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.reward_buffer = np.load(f)
		with gzip.open(path + '$store$_terminal_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.terminal_buffer = np.load(f)
		
		if self.lr is not None:
			l, r = self.lr
			self.observation_buffer = self.observation_buffer[l: r]
			self.action_buffer = self.action_buffer[l: r]
			self.reward_buffer = self.reward_buffer[l: r]
			self.terminal_buffer = self.terminal_buffer[l: r]
		
		self.terminal_idx = list(np.where(self.terminal_buffer == 1)[0])
		self.buffer_length = self.observation_buffer.shape[0]
		print('GameLoader:_load_buffer done. with buffer_length = %d' % self.buffer_length)
	
	def _idx_map(self):
		self.length = 0
		last_timestep = 0
		for i in range(self.buffer_length):
			if i > self.terminal_idx[-1]:
				break
			if last_timestep >= config.history - 1 and (self.terminal_buffer[i: i + config.T] == 0).all():
				self.idx_dict[self.length] = i
				self.length += 1
			last_timestep = 0 if self.terminal_buffer[i] else last_timestep + 1
		print('GameLoader.length = %d' % self.length)
	
	def _reward_delay(self):
		v = 0.
		for i in range(self.buffer_length - 1, -1, -1):
			if i > self.terminal_idx[-1]:
				continue
			if self.terminal_buffer[i]:
				v = -10.
			else:
				# v = config.gamma * v + self.reward_buffer[i]
				v = self.reward_buffer[i]
			self.reward_buffer[i] = v
	
	def _preprocess(self):
		self._idx_map()
		self._reward_delay()
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, idx):
		i = self.idx_dict[idx]
		observation = []
		for j in range(i, i + config.T + 1):
			observation.append(torch.tensor(self.observation_buffer[j + 1 - config.history: j + 1].copy()))
		observation = torch.stack(observation).type(torch.float32) / 255
		action = torch.tensor(self.action_buffer[i: i + config.T + 1].copy(), dtype=torch.long)
		reward = torch.tensor(self.reward_buffer[i: i + config.T + 1].copy(), dtype=torch.float32)
		return observation, action, reward


def main():
	loader = DataLoader(GameLoader(1, 0), batch_size=32, num_workers=12, shuffle=True)
	for observations, actions, rewards in loader:
		print(observations.shape)
		print(actions.shape)
		print(rewards.shape)
		exit(0)


if __name__ == '__main__':
	main()
