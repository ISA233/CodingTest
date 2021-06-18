from torch import nn


class ResBlock(nn.Module):
	def __init__(self, channels):
		super(ResBlock, self).__init__()
		self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.bn0 = nn.BatchNorm2d(channels)
		self.bn1 = nn.BatchNorm2d(channels)
		# self.bn0 = nn.GroupNorm(8, channels)
		# self.bn1 = nn.GroupNorm(8, channels)
		self.act = nn.LeakyReLU()
	
	def forward(self, x):
		c = x
		x = self.bn0(x)
		x = self.act(x)
		x = self.conv0(x)
		x = self.bn1(x)
		x = self.act(x)
		x = self.conv1(x)
		x = x + c
		return x


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding):
		super(ConvBlock, self).__init__()
		self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
		self.bn = nn.BatchNorm2d(out_channels)
		self.act = nn.LeakyReLU()
	
	def forward(self, x):
		x = self.conv0(x)
		x = self.bn(x)
		x = self.act(x)
		return x
