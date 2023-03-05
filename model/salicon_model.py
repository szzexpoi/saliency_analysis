import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import numpy as np
from torch.autograd import Variable
from resnet_ori import resnet50

class SALICON(nn.Module):
	""" A reimplementation of the SALICON saliency prediction model.
	"""
	def __init__(self, embedding_size, use_proto=False, 
				num_proto=1000, second_phase=False, use_interaction=False):
		super(SALICON, self).__init__()
		self.second_phase = second_phase
		self.use_interaction = use_interaction
		self.dilated_backbone = resnet50(pretrained=True)
		self.dilate_resnet(self.dilated_backbone)
		self.dilated_backbone = nn.Sequential(*list(
								self.dilated_backbone.children())[:-2])

		self.v_encoder = nn.Conv2d(2048, 256, kernel_size=3, 
								padding='same', stride=1, bias=True)

		if not self.use_interaction:
			if not second_phase:
				self.sal_layer = nn.Conv2d(512, 1, kernel_size=3, 
										padding='same', stride=1, bias=False)
			else:
				self.sal_layer_ = nn.Conv2d(num_proto, 1, kernel_size=3, 
										padding='same', stride=1, bias=False)
		else:
			# self-attention for probing interactions
			self.sal_query = nn.Linear(num_proto, 256)
			self.sal_key = nn.Linear(num_proto, 256)		
			self.sal_layer_= nn.Conv2d(num_proto, 1, kernel_size=1, 
							padding="same", stride=1, bias=False) 

		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

		self.use_proto = use_proto
		if self.use_proto:
			self.prototype = nn.Linear(512, num_proto, bias=False)
			self.num_proto = num_proto

		for module in [self.dilated_backbone, self.v_encoder, self.prototype]:
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

		image_coarse = F.interpolate(image,(240, 320)) 
		x_fine = self.dilated_backbone(image)
		x_fine = self.v_encoder(x_fine)
		x_coarse = self.dilated_backbone(image_coarse)
		x_coarse = self.v_encoder(x_coarse)
		x_coarse = self.upsample(x_coarse)
		x = torch.cat([x_coarse, x_fine], dim=1)

		if self.use_proto:
			batch, c, h, w = x.shape
			x = x.view(batch, c, h*w)
			proto_sim = torch.sigmoid(self.prototype(x.transpose(1, 2)))

			if self.use_interaction:
				# Transformer-based attention
				query = self.sal_query(proto_sim)
				key = self.sal_key(proto_sim)
				self_att = torch.bmm(query, key.transpose(1, 2))
				self_att = F.softmax(self_att, dim=-1)
				x = torch.bmm(self_att, proto_sim)
			else:
				if not self.second_phase:
					x = torch.bmm(proto_sim, self.prototype.weight.unsqueeze(0).expand(
								batch, self.num_proto, 512))
				else:
					x = proto_sim
			x = x.transpose(1, 2).view(batch, -1, h, w)

			if probe:
				if not self.use_interaction:
					# for analysis, only extracting prototype activation
					return proto_sim.transpose(1, 2).view(batch, -1, h, w)
				else:
					# for analysis of interaction
					proto_sim = proto_sim.transpose(1, 2)
					return proto_sim, self_att

		if not self.second_phase:
			x = torch.sigmoid(self.sal_layer(x))
		else:
			x = torch.sigmoid(self.sal_layer_(x))
		x = F.interpolate(x, (480, 640))		
		return x