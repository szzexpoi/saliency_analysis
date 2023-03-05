import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resnet_multi import resnet50
from vision_transformer import Trans_encoder

class TranSalNet(nn.Module):
	"""
	Reimplementation of TranSalNet (the originally one is hard to converge)
	"""
	def __init__(self, use_proto=False, num_proto=512):
		super(TranSalNet, self).__init__()

		self.backbone = resnet50(pretrained=True)
		self.trans_1 = Trans_encoder(2048, project_dim=768,num_layers=2,
								num_heads=12,num_patches=80,dropout=0) # 80
		self.trans_2 = Trans_encoder(1024, project_dim=768,num_layers=2,
								num_heads=12,num_patches=300,dropout=0) # 300
		self.trans_3 = Trans_encoder(512, project_dim=512,num_layers=2,
								num_heads=8,num_patches=1200,dropout=0) # 1200

		self.sal_decoder_1 = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding='same'),
											nn.BatchNorm2d(768),
											nn.ReLU(),
											nn.Upsample(scale_factor=(2, 2)))

		self.sal_decoder_2 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, stride=1, padding='same'),
											nn.BatchNorm2d(512),
											nn.ReLU(),
											nn.Upsample(scale_factor=(2, 2)))

		self.sal_decoder_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
											nn.BatchNorm2d(512),
											nn.ReLU(),
											nn.Upsample(scale_factor=(2, 2)))

		self.sal_decoder_4 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding='same'),
											nn.BatchNorm2d(128),
											nn.ReLU(),
											nn.Upsample(scale_factor=(2, 2)),
											nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
											nn.BatchNorm2d(128),
											nn.ReLU()
											)

		# self.sal_cls = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding='same', bias=False)
		self.sal_cls_1 = nn.Conv2d(num_proto, 1, kernel_size=3, stride=1, padding='same', bias=False)

		self.use_proto = use_proto
		if self.use_proto:
			self.prototype = nn.Linear(128, num_proto, bias=False)
			self.num_proto = num_proto

		# for fine-tuning experiments
		for module in [self.backbone, self.trans_1, self.trans_2, self.trans_3,
					 self.sal_decoder_1, self.sal_decoder_2, self.sal_decoder_3,
					 self.sal_decoder_4, self.prototype]:
			for para in module.parameters():
				para.requires_grad = False

	def forward(self, image, probe=False):
		# multi-scale features
		x_8, x_16, x_32 = self.backbone(image)

		# saliency decoders with Transformers
		b, c, h, w = x_32.shape
		x_32 = x_32.view(b, c, h*w).transpose(1,2)
		x = self.trans_1(x_32)
		x = x.transpose(1, 2).view(b, -1, h, w)
		x = self.sal_decoder_1(x)
		x = F.interpolate(x, (15, 20))

		b, c, h, w = x_16.shape
		x_16 = x_16.view(b, c, h*w).transpose(1,2)
		x_ = self.trans_2(x_16)
		x_ = x_.transpose(1, 2).view(b, -1, h, w)
		x = x*x_
		x = torch.relu(x)
		x = self.sal_decoder_2(x)

		b, c, h, w = x_8.shape
		x_8 = x_8.view(b, c, h*w).transpose(1,2)	
		x_ = self.trans_3(x_8)
		x_ = x_.transpose(1, 2).view(b, -1, h, w)
		x = x*x_
		x = torch.relu(x)
		x = self.sal_decoder_3(x)
		x = self.sal_decoder_4(x)

		if self.use_proto:
			batch, c, h, w = x.shape

			x = x.view(batch, c, h*w)
			proto_sim = torch.sigmoid(self.prototype(x.transpose(1, 2)))

			# x = torch.bmm(proto_sim, self.prototype.weight.unsqueeze(0).expand(
			# 			batch, self.num_proto, 128))

			x = proto_sim
			x = x.transpose(1, 2).view(batch, -1, h, w)

			if probe:
				return proto_sim.transpose(1, 2).view(batch, -1, h, w)

		# x = self.sal_cls(x)
		x = self.sal_cls_1(x) # for fine-tuning (analysis)
		x = F.interpolate(x, (240, 320))
		x = torch.sigmoid(x)

		return x
