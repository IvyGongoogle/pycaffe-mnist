#!/usr/bin/env python
#coding:utf-8

import os.path
import numpy as np
from config import cfg

class imdb(object):
	
	def __init__(self, phase='train'):
		##############################
		#         可配置参数          #
		##############################
		self._data_dir = 'data'
		self._ext = '.png'
		##############################
		self.phase = phase
		label_file = os.path.join(self._data_dir, phase + '_label.txt')
		with open(label_file, 'r') as f:
			lines = f.readlines()
		
		num_imgs = len(lines)
		print 'num_imgs: ', num_imgs
		self.im_nums = num_imgs
		self._label = np.zeros((num_imgs, 1), dtype='int32')
		self._name_to_id = {}
		self._id_to_name = {}
		
		for im_id in xrange(num_imgs):
			# im_name = str(i)
			# cls = int(lines[i].strip())  # 类别class缩写为cls
			# self._label[i] = cls
			# self._name_to_inds[im_name] = i
			im_name, im_label = lines[im_id].strip().split(' ')
			self._label[im_id] = im_label
			self._name_to_id[im_name] = im_id
			self._id_to_name[im_id] = im_name
	
	def path(self, im_id):
		"""
		根据图像id，获取图片存储路径（绝对路径）
		"""
		im_name = self._id_to_name[im_id]
		im_path = os.path.join(self._data_dir, im_name + self._ext)
		return im_path

	def label(self, im_id):
		"""
		根据图像id，获取其label
		"""
		return self._label[im_id]
	
	def path_by_name(self, im_name):
		"""
		根据图像名字，获取图片存储路径
		"""
		im_path = os.path.join(self._data_dir, self.phase, im_name + self._ext)
		return im_path

	def label_by_name(self, im_name):
		"""
		根据图像名字，获取图片label
		"""
		im_id = self._name_to_id[im_name]
		return self._label[im_id]
	
