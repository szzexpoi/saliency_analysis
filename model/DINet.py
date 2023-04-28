import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
from torch.autograd import Variable
from resnet_ori import resnet50


class Inception_Encoder(nn.Module):
	""" An inception-like encoder for saliency prediction.
	"""
	def __init__(self, input_size, embedding_size):
		super(Inception_Encoder, self).__init__()
		self.inception_1 = nn.Conv2d(input_size, embedding_size, kernel_size=1,
								padding="same", stride=1, dilation=1, bias=False)
		self.inception_2 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
								padding="same", stride=1, dilation=4, bias=False)
		self.inception_3 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
								padding="same", stride=1, dilation=8, bias=False)
		self.inception_4 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
								padding="same", stride=1, dilation=16, bias=False)

	def forward(self, x):
		""" Implementation of the inception data flow proposed
			in the DINet paper. Note that three of the branches
			are conditioned on the first conv layer, and there is
			a sum fusion along side the independent branches.

			Input:
				x: A Batch x N x H x W tensor encoding the visual
				features extracted from the backbone, where N is
				the number of filters for the features.
			Return:
				A Batch x M x H x W tensor encoding the features
				processed by the inception encoder, where M is the embedding
				size*4.

		"""
		x = torch.relu(self.inception_1(x))
		b_1 = torch.relu(self.inception_2(x))
		b_2 = torch.relu(self.inception_3(x))
		b_3 = torch.relu(self.inception_4(x))
		fused_b = b_1 + b_2 + b_3 # sum fusion

		return torch.cat([fused_b, b_1, b_2, b_3], dim=1)


class Simple_Decoder(nn.Module):
	""" A simple feed-forward decoder for saliency prediction.
	"""

	def __init__(self, input_size, embedding_size):
		super(Simple_Decoder, self).__init__()
		self.decoder_1 = nn.Conv2d(input_size, embedding_size, kernel_size=3,
						padding="same", stride=1, bias=False)
		self.decoder_2 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
						padding="same", stride=1, bias=False)
		self.decoder_3= nn.Conv2d(embedding_size, 1, kernel_size=3,
						padding="same", stride=1, bias=False)

	def forward(self, x):
		""" A standard feed-forward flow of decoder.
			Note that at the end there is a rescaling
			operation.
		"""

		x = torch.relu(self.decoder_1(x))
		x = torch.relu(self.decoder_2(x))
		x = torch.sigmoid(self.decoder_3(x))
		x = F.interpolate(x,(480, 640))

		return x

class Proto_Decoder(nn.Module):
	""" Decoder with factorized prototypes
	"""

	def __init__(self, input_size, embedding_size,
				num_proto=1000, second_phase=False):
		super(Proto_Decoder, self).__init__()
		self.num_proto = num_proto
		self.embedding_size = embedding_size
		self.second_phase = second_phase
		self.v_decoder = nn.Conv2d(input_size, embedding_size, kernel_size=3,
						padding="same", stride=1, bias=False)
		# projecting visual features onto the probability of prototypes
		self.prototype = nn.Linear(embedding_size, num_proto, bias=False)

		if not second_phase:
			self.sal_decoder= nn.Conv2d(embedding_size, 1, kernel_size=3,
							padding="same", stride=1, bias=False)
		else:
			# use a different layer name for partial weight loading
			self.sal_decoder_1= nn.Conv2d(num_proto, 1, kernel_size=3,
							padding="same", stride=1, bias=False)

		# for fine-tuning (second phase)
		if second_phase:
			for module in [self.v_decoder, self.prototype]:
				for para in module.parameters():
					para.requires_grad = False


	def forward(self, x, probe=False):
		""" A standard feed-forward flow of decoder.
			Note that at the end there is a rescaling
			operation.
		"""

		x = torch.relu(self.v_decoder(x))
		batch, c, h, w = x.shape
		x = x.view(batch, c, h*w)
		proto_sim = torch.sigmoid(self.prototype(x.transpose(1, 2)))

		if not self.second_phase:
			x = torch.bmm(proto_sim, self.prototype.weight.unsqueeze(0).expand(
								batch, self.num_proto, self.embedding_size))
		else:
			x = proto_sim

		x = x.transpose(1, 2).view(batch, -1, h, w)

		if probe:
			# for analysis, only extracting prototype activation
			return proto_sim.transpose(1, 2).view(batch, -1, h, w)

		else:
			if not self.second_phase:
				x = torch.sigmoid(self.sal_decoder(x))
			else:
				x = torch.sigmoid(self.sal_decoder_1(x))
			x = F.interpolate(x,(480, 640)) # for SALICON

			return x


class DINet(nn.Module):
	""" A reimplementation of the saliency prediction model
		introduced in the following paper:
		https://arxiv.org/abs/1904.03571
	"""
	def __init__(self, embedding_size, use_proto=False,
				num_proto=1000, second_phase=False):
		super(DINet, self).__init__()
		self.dilated_backbone = resnet50(pretrained=True)
		self.dilate_resnet(self.dilated_backbone) # DINet use the same Dilated ResNet as SAM
		self.dilated_backbone = nn.Sequential(*list(
								self.dilated_backbone.children())[:-2])
		self.encoder = Inception_Encoder(2048, embedding_size)
		if not use_proto:
			self.decoder = Simple_Decoder(embedding_size*4, embedding_size)
		else:
			self.decoder = Proto_Decoder(embedding_size*4,
									embedding_size, num_proto, second_phase)

		if second_phase:
			for module in [self.dilated_backbone, self.encoder]:
				for para in module.parameters():
					para.requires_grad = False

	def dilate_resnet(self, resnet):
		""" Converting standard ResNet50 into a dilated one.
		"""
		resnet.layer3[0].conv1.stride = 1
		resnet.layer3[0].downsample[0].stride = 1
		resnet.layer4[0].conv1.stride = 1
		resnet.layer4[0].downsample[0].stride = 1

		for block in resnet.layer3:
			block.conv2.dilation = 2
			block.conv2.padding = 2

		for block in resnet.layer4:
			block.conv2.dilation = 4
			block.conv2.padding = 4

	def forward(self, image, probe=False):
		""" Data flow for DINet. Most of the key
			components have been implemented in
			separate modules.
		"""

		x = self.dilated_backbone(image)
		x = self.encoder(x)
		x = self.decoder(x, probe)

		return x
