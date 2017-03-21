#!/usr/bin/env python
#coding:utf-8

import caffe
import numpy as np
import cv2
from config import cfg
import lmdb
from caffe.proto import caffe_pb2
import time

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
		# start = time.time()
		data_blob, label_blob = self._get_next_minibatch()
		# print('python layer fetching minibatch consumes {:f} seconds'.format(time.time()-start))
		# _get_next_minibatch()耗时0.008214秒，基本上时间开销都在这里了
		
		top[0].reshape(*(data_blob.shape))
		top[0].data[...] = data_blob
		
		top[1].reshape(*(label_blob.shape))
		top[1].data[...] = label_blob
		# print('python layer doing forward consumes {:f} seconds'.format(time.time()-start))
		# 整个forward耗时0.008247秒
	
	def backward(self, bottom, top):
		"""本层不需要实现"""
		pass
		
	def reshape(self, bottom, top):
		"""本层不需要实现"""
		pass
	
	def prepare_imdb(self, imdb):
		# self.imdb = imdb
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
		
		txn = self.db.begin()
		cursor = txn.cursor()
		datum = caffe_pb2.Datum()

		
		for im_id in xrange(self._cur, self._cur+self._batch_size):

			color = True
			if im_C == 1:
				color = False
			# im_caffe = caffe.io.load_image(self.imdb.path(im_id), color)
			keystr = '{:0>8d}'.format(im_id)
			value = txn.get(keystr)
			datum.ParseFromString(value)
			label = datum.label
			# data = caffe.io.datum_to_array(datum)   # CxHxW   # 耗时0.000073，需要优化
			# data = np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
			data=np.array(datum.float_data).astype(float).reshape(datum.channels, datum.height, datum.width)
			im = np.transpose(data, (1,2,0))    # CxHxW -> HxWxC in cv2
			# data = np.fromstring(datum.data, dtype=np.uint8).reshape(datum.height, datum.width)
			# im = np.transpose(data, (1,2,0))
			# print('im_caffe\'s dtype is', im_caffe.dtype, ' im\'s dtype is', im.dtype)


			blob_id = im_id - self._cur

			data_blob[blob_id,...] = im[...]

			# imdb_label = self.imdb.label(im_id)
			
			the_label = np.array([label], dtype=np.int32)

			#print('imdb_label is', imdb_label, ' the_label is', the_label)

			label_blob[blob_id] = the_label

			#cv2.imshow('图', im)
			#cv2.waitKey(0)
			#print 'label is: ', self.imdb.label(im_name)

		self._cur = self._cur + self._batch_size
		#data_blob = data_blob.transpose((2,1,3,0))
		data_blob = data_blob.transpose((0, 3, 1, 2))
		return data_blob, label_blob
