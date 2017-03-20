#!/usr/bin/env python
#coding:utf-8

import caffe
import numpy as np
import cv2
from config import cfg
import lmdb

DEBUG = False
class MnistLmdbDataLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		self._batch_size = cfg.TRAIN.BATCH_SIZE
		self._cur = 0
		
		top[0].reshape(self._batch_size, 1, 28, 28)
		top[1].reshape(self._batch_size, 1)

		self.db = lmdb.open(cfg.TRAIN.TRAIN_LMDB)
		
	def forward(self, bottom, top):
		"""
		获取一个minibatch的blob数据，作为网络的输入。bottom参数忽略即可
		top[0]: data
		top[1]: label
		"""
		data_blob, label_blob = self._get_next_minibatch()
		
		top[0].reshape(*(data_blob.shape))
		top[0].data[...] = data_blob
		
		top[1].reshape(*(label_blob.shape))
		top[1].data[...] = label_blob
	
	def backward(self, bottom, top):
		"""本层不需要实现"""
		pass
		
	def reshape(self, bottom, top):
		"""本层不需要实现"""
		pass
	
	def prepare_imdb(self, imdb):
		self.imdb = imdb
		self.im_nums = imdb.im_nums
		assert self.im_nums > self._batch_size, 'batch_size设定为{:d}。图片数量至少是64,最好是64的整数倍'.format(self._batch_size)
		
	def _get_next_minibatch(self):
		"""
		真正读取一个minibatch数据的函数：image+txt -> blob
		返回一个blob，尺寸为(batch_size, 1, 图高，图宽)
		"""
		loaded_ims = []
		im_H = 28  # im height
		im_W = 28  # im width
		im_C = 1   # im channel
		data_blob = np.zeros((self._batch_size, im_H, im_W, im_C), dtype=np.float32)
		label_blob = np.zeros((self._batch_size, 1), dtype=np.int32)   # TODO:尝试支持多标签？
		if self._cur+self._batch_size > self.im_nums:
			self._cur = 0
		if DEBUG:
			print '\nself._cur = ', self._cur
			print 'self._batch_size = ', self._batch_size
			print 'self.im_nums = ', self.im_nums
		
		txn = db.begin()
		cursor = txn.cursor()
		datum = caffe_pb2.Datum()
		for im_id in xrange(self._cur, self._cur+self._batch_size):
			color = True
			if im_C == 1:
				color = False
			im = caffe.io.load_image(self.imdb.path(im_id), color)
			blob_id = im_id - self._cur
			
			data_blob[blob_id,...] = im[...]

			label_blob[blob_id] = self.imdb.label(im_id)
			#cv2.imshow('图', im)
			#cv2.waitKey(0)
			#print 'label is: ', self.imdb.label(im_name)
		self._cur = self._cur + self._batch_size
		#data_blob = data_blob.transpose((2,1,3,0))
		data_blob = data_blob.transpose((0, 3, 1, 2))
		return data_blob, label_blob
