import numpy as np
import random
import os
import time
import torch
import torch.utils.data as data
import json
from PIL import Image
from torchvision import transforms
import cv2
import pandas as pd
from nltk.stem import WordNetLemmatizer

class salicon(data.Dataset):
    def __init__(self, anno_dir, fix_dir, img_dir, width, 
    			height, mode='train', transform=None):
        self.anno_data = glob(os.path.join(anno_dir, mode ,'*.jpg'))
        self.fix_dir = os.path.join(fix_dir, mode)
        self.img_dir = os.path.join(img_dir, mode)
        self.width = width
        self.height = height
        self.transform = transform
        self.mode = mode

    def get_fixation(self,fix_data):
        fix_data = loadmat(fix_data)

        #loading new salicon data
        fixation_map = fix_data['fixationPts'].astype('float32')

        return fixation_map

    def __getitem__(self,index):
        cur_id = os.path.basename(self.anno_data[index])[:-4]
        # loading the saliency map
        cur_anno = cv2.imread(self.anno_data[index]).astype('float32')
        cur_anno = cur_anno[:,:,0]
        cur_anno = cv2.resize(cur_anno,(self.width,self.height))
        cur_anno /= cur_anno.max()
        cur_anno = torch.from_numpy(cur_anno)

        # loading the image
        cur_img = Image.open(os.path.join(self.img_dir, cur_id+'.jpg')).convert('RGB')
        if self.transform is not None:
            cur_img = self.transform(cur_img)

        # loading the fixation map
        cur_fix = self.get_fixation(os.path.join(self.fix_dir, cur_id+'.mat'))
        cur_fix = cv2.resize(cur_fix,(self.width, self.height))
        cur_fix[cur_fix>0.1] = 1
        cur_fix[cur_fix!=1]= 0

        return cur_img, cur_anno, torch.from_numpy(cur_fix), cur_id

    def __len__(self,):
        return len(self.anno_data)

class VG_generator(data.Dataset):
	""" Data loader for processing the Visual Genome
		dataset for prototype dissection.
	"""
	def __init__(self, img_dir, data_dir):
		self.data_dir = data_dir
		self.img_dir = img_dir
		self.annotation = json.load(open(os.path.join(data_dir, 'train_sceneGraphs.json')))
		self.vg2coco = json.load(open('coco_idx.json'))
		self.filtered_obj = json.load(open('filtered_vg_obj.json'))
		self.filtered_attr = json.load(open('filtered_vg_attr.json'))
		
		# define the data augmentation for image
		self.transform = transforms.Compose([
				transforms.Resize((480, 640)), # for SALICON, DINet
				# transforms.Resize((240, 320)), # for transalnet
				transforms.ToTensor(), 
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		self.lemmatizer = WordNetLemmatizer() # need to lemmatize VG labels

		self.init_data()


	def init_data(self,):
		# Initialize the pool of data
		self.img_pool = []
		self.concept_seg = dict()
		self.img_size = dict()
		self.concept_pool = dict()

		# reorganize the data
		for img_id in self.annotation:
			img_w = self.annotation[img_id]['width']
			img_h = self.annotation[img_id]['height']
			self.img_size[img_id] = [int(img_h), int(img_w)]

			# get dense annotations (excluding relationships)
			tmp_data = dict()
			for obj_id in self.annotation[img_id]['objects']:
				# filter out non-overlapping objects
				obj_name = self.annotation[img_id]['objects'][obj_id]['name']
				obj_name = '_'.join(obj_name.split(' '))
				obj_name = self.lemmatizer.lemmatize(obj_name)

				# select objects either exist in COCO or are of high frequencies
				if obj_name not in self.vg2coco and not self.filtered_obj[obj_name]:
					continue

				# bbox of the object
				x = int(self.annotation[img_id]['objects'][obj_id]['x'])
				y = int(self.annotation[img_id]['objects'][obj_id]['y'])
				w = int(self.annotation[img_id]['objects'][obj_id]['w'])
				h = int(self.annotation[img_id]['objects'][obj_id]['h'])

				# jointly considering the effects of objects and their attributes

				if obj_name not in tmp_data:
					tmp_data[obj_name] = []
				tmp_data[obj_name].append([x, y, w, h])

				if obj_name not in self.concept_pool:
					self.concept_pool[obj_name] = 1
				else:
					self.concept_pool[obj_name] += 1

				# attributes
				for attr in self.annotation[img_id]['objects'][obj_id]['attributes']:
					if not self.filtered_attr[attr]:
						continue
					if attr not in tmp_data:
						tmp_data[attr] = []
					tmp_data[attr].append([x, y, w, h])
					if not attr in self.concept_pool:
						self.concept_pool[attr] = 1
					else:
						self.concept_pool[attr] += 1

			if len(tmp_data)>0:
				self.img_pool.append(img_id)
				self.concept_seg[img_id] = tmp_data


	def decode_vg(self, img_id):
		# decode the segmentation data for a single image in Visual Genome
		segmentation_mask = dict()
		img_h, img_w = self.img_size[img_id]
		for concept in self.concept_seg[img_id]:
			concept_mask = np.zeros([img_h, img_w]).astype('float32')
			for loc in self.concept_seg[img_id][concept]:
				x, y, w, h = loc
				concept_mask[y:y+h, x:x+w] = 1
			concept_mask = cv2.resize(concept_mask, (80, 60)) # for SALICON and DINet
			# concept_mask = cv2.resize(concept_mask, (160, 120)) # for TranSalNet
			concept_mask[concept_mask>0.9] = 1
			concept_mask[concept_mask<1] = 0
			segmentation_mask[concept] = concept_mask

		return segmentation_mask


	def __getitem__(self, index):
		img_id = self.img_pool[index]
		img = Image.open(os.path.join(self.img_dir, img_id+'.jpg')).convert('RGB')
		img = self.transform(img)

		return img, [img_id]

	def __len__(self,):
		return len(self.img_pool)