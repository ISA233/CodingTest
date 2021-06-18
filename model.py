import numpy as np
import torch
from torch import nn
from data import DataLoader
from config import config
from torch.nn import functional as F
from torch import optim
from utils import save_img_pair
from base import ResBlock


class Repr(nn.Module):
	def __init__(self):
		super(Repr, self).__init__()
		self.conv0 = nn.Conv2d(config.history, 32, kernel_size=1)
		self.res0 = ResBlock(32)
		self.res1 = ResBlock(32)
		
		self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.res2 = ResBlock(64)
		self.res3 = ResBlock(64)
		
		self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.res4 = ResBlock(128)
		self.res5 = ResBlock(128)
		
		self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.res6 = ResBlock(256)
		self.res7 = ResBlock(256)
		
		self.linear = nn.Linear(256 * 9, config.repr_num)
		self.value = nn.Linear(config.repr_num, 1, bias=False)
		self.policy = nn.Linear(config.repr_num, 4, bias=False)
		self.act = nn.LeakyReLU()
		self.pool = nn.MaxPool2d(2)
	
	def forward(self, observations: torch.Tensor):
		shape = observations.shape
		if len(shape) == 5:
			observations = observations.view([shape[0] * shape[1], *shape[2:]])
		x = observations
		
		x = self.conv0(x)
		x = self.res0(x)
		x = self.res1(x)
		x = self.pool(x)
		x = self.conv1(x)
		
		x = self.res2(x)
		x = self.res3(x)
		x = self.pool(x)
		x = self.conv2(x)
		
		x = self.res4(x)
		x = self.res5(x)
		x = self.pool(x)
		x = self.conv3(x)
		
		x = self.res6(x)
		x = self.res7(x)
		
		z = x.view(x.shape[0], -1)
		z = self.linear(z)
		z = self.act(z)
		p = self.policy(z)
		v = self.value(z)
		
		if len(shape) == 5:
			return z.view(*shape[:2], -1), p.view(*shape[:2], -1), v.view(*shape[:2])
		return z, p, v
	
	@staticmethod
	def loss(p, v, actions, rewards):
		ce = nn.CrossEntropyLoss()
		mse = nn.MSELoss()
		
		v_loss = mse(v[:, :-1], rewards[:, :-1] + config.gamma * v[:, 1:])
		p_loss = ce(p.view(-1, 4), actions.view(-1))
		return v_loss, p_loss


class Upsample(nn.Module):
	def __init__(self):
		super(Upsample, self).__init__()
	
	def forward(self, x):
		return F.interpolate(x, scale_factor=2)


class Inverse(nn.Module):
	def __init__(self):
		super(Inverse, self).__init__()
		self.linear = nn.Linear(config.repr_num, 256 * 9)
		self.res0 = ResBlock(256)
		self.res1 = ResBlock(256)
		self.conv0 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
		
		self.res2 = ResBlock(128)
		self.res3 = ResBlock(128)
		self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		
		self.res4 = ResBlock(64)
		self.res5 = ResBlock(64)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
		
		self.res6 = ResBlock(32)
		self.res7 = ResBlock(32)
		self.conv3 = nn.Conv2d(32, 1, kernel_size=1)
		
		self.act = nn.LeakyReLU()
		self.up = Upsample()
	
	def forward(self, x):
		# print('inverse', x.shape)
		x = self.linear(x)
		x = self.act(x)
		x = x.view(-1, 256, 3, 3)
		
		x = self.res0(x)
		x = self.res1(x)
		x = self.conv0(x)
		x = self.up(x)
		
		x = self.res2(x)
		x = self.res3(x)
		x = self.conv1(x)
		x = self.up(x)
		
		x = self.res4(x)
		x = self.res5(x)
		x = self.conv2(x)
		x = self.up(x)
		
		x = self.res6(x)
		x = self.res7(x)
		x = self.conv3(x)
		# x = torch.sigmoid(x)
		return x[:, 0]


class Trans(nn.Module):
	def __init__(self):
		super(Trans, self).__init__()
		self.linear0 = nn.Linear(config.repr_num, config.repr_num, bias=False)
		self.linear1 = nn.Linear(4, config.repr_num)
		self.act = nn.LeakyReLU()
		self.linear2 = nn.Linear(config.repr_num, config.repr_num)
	
	def forward(self, z, actions):
		# print('Z', z.shape)
		zs = [z.clone()]
		for t in range(config.T):
			z = self.linear0(z) + self.linear1(actions[:, t])
			z = self.act(z)
			z = self.linear2(z)
			z = self.act(z)
			zs.append(z.clone())
		z = torch.stack(zs, dim=1)
		# print('Zs', z.shape)
		return z


class Map(nn.Module):
	def __init__(self):
		super(Map, self).__init__()
		self.linear0 = nn.Linear(config.repr_num, config.repr_num, bias=False)
		self.act = nn.LeakyReLU()
		self.linear1 = nn.Linear(config.repr_num, config.repr_num, bias=False)
	
	def forward(self, x):
		x = self.linear0(x)
		x = self.act(x)
		x = self.linear1(x)
		return x


class SPR(nn.Module):
	def __init__(self):
		super(SPR, self).__init__()
		self.name = 'spr'
		self.repr = Repr()
		self.trans = Trans()
		self.map = Map()
		self.inverse = Inverse()
		self.optim = optim.SGD(self.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.wd)
	
	def forward(self, x):
		pass
	
	def save(self):
		torch.save(self.state_dict(), 'save/%s.pkl' % self.name)
	
	def restore(self):
		self.load_state_dict(torch.load('save/%s.pkl' % self.name))
	
	def learn(self, observations, actions, rewards):
		self.train()
		# self.eval()  # TODO eval !!!!!!!!!!!!!!!!!!!!!!!!!!
		
		z, p, v = self.repr(observations)
		# vp_loss
		v_loss, p_loss = self.repr.loss(p, v, actions, rewards)  # currently T + 1 times loss calc every epoch
		# ~
		
		with torch.no_grad():
			acc = (torch.argmax(p, dim=-1) == actions).type(torch.float32).mean()
			print(acc.item() * 100)
		
		actions = actions[:, :-1]
		actions_one_hot = torch.zeros([*actions.shape, 4], dtype=torch.float32, device=actions.device)
		actions_one_hot = actions_one_hot.scatter_(2, actions.view(*actions.shape, 1), 1.)
		
		# spr_loss
		z_hat = self.trans(z[:, 0], actions_one_hot)

		eps = 1e-5
		y = self.map(z).view(-1, config.repr_num)
		# print(y.shape, y.norm(dim=1).shape)
		y = y / (y.norm(dim=1) + eps).view(-1, 1)
		y_hat = self.map(z_hat).view(-1, config.repr_num)
		y_hat = y_hat / (y_hat.norm(dim=1) + eps).view(-1, 1)
		spr_ali_loss = (y - y_hat).norm(dim=1).pow(2).mean()
		spr_uni_loss_0 = (torch.pdist(y).pow(2).mul(-2)).exp().mean().log()
		spr_uni_loss_1 = (torch.pdist(y_hat).pow(2).mul(-2)).exp().mean().log()
		spr_loss = spr_ali_loss + spr_uni_loss_0 + spr_uni_loss_1
		# print('%.5f %.5f %.5f' % (spr_ali_loss.item(), spr_uni_loss_0.item(), spr_uni_loss_1.item()))
		# spr_loss = torch.tensor(0.)

		# # obs_loss
		i = self.inverse(z_hat)
		j = observations[:, :, -1].contiguous().view(-1, 24, 24)  # TODO it is [B, T] !!!!
		obs_loss = ((i - j) ** 2).sum(dim=(1, 2)).mean()
		print(i.shape, j.shape)

		with torch.no_grad():
			for _ in range(config.T + 1):
				a = j[_].cpu()
				b = i[_].cpu()
				try:
					save_img_pair(a, b, _)
				except Exception as e:
					print('save_img failed.')
					print(e)
		# obs_loss = torch.tensor(0.)
		
		self.optim.zero_grad()
		loss = v_loss * config.v_loss + \
		       p_loss + \
		       spr_loss + \
		       obs_loss * config.obs_loss
		# loss = obs_loss
		loss.backward()
		nn.utils.clip_grad_norm_(self.parameters(), max_norm=config.clip)
		self.optim.step()
		return v_loss.item(), p_loss.item(), spr_loss.item(), obs_loss.item()
		# return 0., 0., 0., obs_loss.item()


class MultigpuSPR(nn.Module):
	def __init__(self, spr: SPR):
		super(MultigpuSPR, self).__init__()
		self.spr = spr
	
	def forward(self, observations, actions, rewards):
		self.train()
		z, v, p = self.spr.repr(observations)
		
		actions = actions[:, :-1]
		
		actions_one_hot = torch.zeros([*actions.shape, 4], dtype=torch.float32, device=actions.device)
		actions_one_hot = actions_one_hot.scatter_(2, actions.view(*actions.shape, 1), 1.)
		
		z_hat = self.spr.trans(z[:, 0], actions_one_hot)
		
		i = self.spr.inverse(z_hat)
		j = observations[:, :, -1].contiguous().view(-1, 24, 24)
		obs_loss = ((i - j) ** 2).sum(dim=(1, 2))
		
		return obs_loss
