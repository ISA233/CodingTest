import torch
import numpy as np

np.set_printoptions(linewidth=9999999)


class Config:
	def __init__(self):
		self.lr = 0.01
		self.wd = 0.0001
		self.repr_num = 512
		self.batch_size = 384
		self.gamma = 0.99
		self.clip = 5.
		
		self.image_size = 24
		
		self.T = 5
		self.history = 8
		
		self.v_loss = 1.
		self.obs_loss = 0.25
		self.device = torch.device('cuda:0')
		self.path = 'C:/Users/YveH/Documents/Breakout/'
		# self.path = '/data/zys/Breakout/'
		
		self.multigpu = False
		self.devices = [0, 1, 2, 3]


config = Config()
