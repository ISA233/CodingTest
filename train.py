from data import GameLoader
from model import Repr, SPR
from config import config
from torch.utils.data import DataLoader
from utils import log, time_str
from torch import nn

if config.multigpu:
	from model import MultigpuSPR


def train_epoch(model, loader):
	cnt = 0
	for observations, actions, rewards in loader:
		if observations.shape[0] != config.batch_size:
			break
		observations = observations.to(config.device)
		actions = actions.to(config.device)
		rewards = rewards.to(config.device)
		v_loss, p_loss, spr_loss, obs_loss = model.learn(observations, actions, rewards)
		cnt += 1
		print('%.6f %.6f %.6f %.6f' % (v_loss, p_loss, -spr_loss, obs_loss))
		if cnt % 100 == 0:
			model.save()
			print('save.')


def train(model):
	for block_id in range(0, 40):
		block_id = 40
		log.write('%s: new block %d\n' % (time_str(), block_id))
		loader = DataLoader(GameLoader(1, block_id), batch_size=config.batch_size, num_workers=12, shuffle=True)
		train_epoch(model, loader)
		loader = None


def multigpu_train_epoch(model, loader):
	cnt = 0
	for observations, actions, rewards in loader:
		if observations.shape[0] != config.batch_size * len(config.devices):
			break
		observations = observations.to(config.device)
		actions = actions.to(config.device)
		rewards = rewards.to(config.device)
		obs_loss = model(observations, actions, rewards).mean()
		
		model.module.spr.optim.zero_grad()
		obs_loss.backward()
		nn.utils.clip_grad_norm_(model.module.spr.parameters(), max_norm=config.clip)
		model.module.spr.optim.step()
		
		obs_loss = obs_loss.item()
		
		cnt += 1
		print('%.6f' % obs_loss)
		if cnt % 100 == 0:
			model.module.spr.save()
			print('save.')


def multigpu_train(model):
	for block_id in range(0, 40):
		log.write('%s: new block %d\n' % (time_str(), block_id))
		loader = DataLoader(GameLoader(1, block_id), batch_size=config.batch_size * len(config.devices), num_workers=16,
		                    shuffle=True)
		multigpu_train_epoch(model, loader)


def main():
	log.set_file()
	model = SPR().to(config.device)
	# model.restore()
	
	if not config.multigpu:
		while True:
			train(model)
	else:
		multigpu_model = nn.DataParallel(MultigpuSPR(model), device_ids=config.devices)
		while True:
			multigpu_train(multigpu_model)


if __name__ == '__main__':
	main()
