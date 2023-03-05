import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict

class MLP(nn.Module):
	""" MLP layer for vision transformer.
	"""
	def __init__(self, input_size, hidden_size=[128,64],dropout=0.1):
		super(MLP, self).__init__()
		module_list = []
		prev_size = input_size
		for i in range(len(hidden_size)):
			module_list.append(('fc_'+str(i+1),
							nn.Linear(prev_size, hidden_size[i])))
			module_list.append(('gelu_'+str(i+1), nn.GELU()))
			module_list.append(('dp_'+str(i+1), nn.Dropout(dropout)))
			prev_size = hidden_size[i]

		self.mlp = nn.Sequential(OrderedDict(module_list))

	def forward(self,x):
		return self.mlp(x)


class PatchEncoder(nn.Module):
	""" Patch encoder for integrating positional
		information with visual features.
	"""
	def __init__(self, num_patches, input_size, project_dim=64):
		super(PatchEncoder, self).__init__()
		self.num_patches = num_patches
		self.projection = nn.Linear(input_size, project_dim)
		self.pos_embedding = nn.Embedding(
									num_embeddings=num_patches,embedding_dim=project_dim)

	def forward(self,x):
		pos = torch.range(start=0, end=self.num_patches-1, step=1).long().cuda()
		encoded = self.projection(x) + self.pos_embedding(pos.unsqueeze(0))
		return encoded


class Trans_encoder(nn.Module):
	""" Function for implementing a single transformer
		block in vision transformer.
	"""
	def __init__(self, input_dim, project_dim=64, num_layers=2, 
				num_heads=8, num_patches=300, dropout=0.1):
		super(Trans_encoder,self).__init__()
		self.num_layers = num_layers
		self.patch_encoder = PatchEncoder(
									int(num_patches), input_dim, project_dim)
		
		# construct the main components of ViT with nested lists
		self.attention = []
		self.norm_before = []
		self.norm_after = []
		self.mlp  = []

		for i in range(num_layers):
			self.attention.append(nn.MultiheadAttention(embed_dim=project_dim, 
								num_heads=num_heads,dropout=dropout, batch_first=True))
			self.norm_before.append(nn.LayerNorm(project_dim, eps=1e-6))
			self.norm_after.append(nn.LayerNorm(project_dim, eps=1e-6))
			self.mlp.append(MLP(input_size=project_dim, 
							hidden_size=[project_dim*2, project_dim], dropout=dropout))

		self.attention = nn.ModuleList(self.attention)
		self.norm_before = nn.ModuleList(self.norm_before)
		self.norm_after = nn.ModuleList(self.norm_after)
		self.mlp = nn.ModuleList(self.mlp)

		self.output_norm = nn.LayerNorm(project_dim, eps=1e-6)

	def forward(self,image):
		# encode image and add position embedding
		x = self.patch_encoder(image)

		# main loop for multi-layer Transformer
		for i in range(self.num_layers):
			x1 = self.norm_before[i](x)
			attention_output, _ = self.attention[i](
									x1, x1, x1, need_weights=False)
			x2 = attention_output+x
			x3 = self.norm_after[i](x2)
			x3 = self.mlp[i](x3)
			x = x3 + x2

		# classifier
		x = self.output_norm(x)

		return x





