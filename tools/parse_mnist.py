#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将mnist数据转化为图片和文本格式的label
数据准备：
	t10k-images-idx3-ubyte
	t10k-labels-idx1-ubyte
	train-images-idx3-ubyte
	train-labels-idx1-ubyte
程序执行后得到：
	训练图像 train/目录
	测试图像 test/目录
	训练标签 train_label.txt
	测试标签 test_label.txt

其中，标签文件中每一行是一个数字，行号对应到图片编号，行内容为标签，也就是0~9里面的一个数字
"""

import _init_paths
from PIL import Image
import struct
import cv2
import numpy as np
import os
from config import cfg

ID_START = 1  # 数据集里面图片起始编号。可以换成0

def read_image(filename, phase):
	"""
	从mnist数据中解析出图片，并存储
	@param phase: train/test
	"""
	f = open(filename, 'rb')

	index = 0
	buf = f.read()

	f.close()
	magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
	index += struct.calcsize('>IIII')
	
  	for i in xrange(images):
		#for i in xrange(1):
  		#for i in xrange(2000):
    	#image = Image.new('L', (columns, rows))
		image = np.zeros((rows, columns, 1), np.uint8)
		
		for x in xrange(rows):
			for y in xrange(columns):
				image[x,y] = int(struct.unpack_from('>B', buf, index)[0])
				index += struct.calcsize('>B')
		
		im_name = str(ID_START + i)
		print 'save ' + im_name + 'image'
		im_save_name = phase + '/%s.png' % im_name
		cv2.imwrite(im_save_name, image)
		
def read_label(filename, phase):
	"""
	从mnist数据中解析出类别标签(label)，并存储到txt文件
	每行格式： 图片名（不含后缀和路径前缀） 图片标签
	"""
	f = open(filename, 'rb')
	index = 0
	buf = f.read()
	
	f.close()
	magic, labels = struct.unpack_from('>II' , buf , index)
	index += struct.calcsize('>II')
	
	# labelArr = [0] * labels
	#labelArr = [0] * 2000
	
	saveFilename = phase + '_label.txt'
	save = open(saveFilename, 'w')

	for i in xrange(labels):
    # for i in xrange(2000):
		# labelArr[i] = int(struct.unpack_from('>B', buf, index)[0])
		label = struct.unpack_from('>B', buf, index)[0]
		index += struct.calcsize('>B')
		im_name = phase + '/' + str(ID_START + i)
		line = '{:s} {:d}'.format(im_name, label)
		save.write(line + '\n')
	
	# save.write('\n'.join(map(lambda x: str(x), labelArr)))
	save.close()
	print 'save labels success'


if __name__ == '__main__':
	#root_dir = 'd:/code/mnist'
	# root_dir = '/home/chris/work/mnist'
	root_dir = cfg.META.ROOT_DIR
	data_dir = os.path.join(root_dir, 'data')
	os.chdir(data_dir)
	
	#read_image('raw_data/train-images-idx3-ubyte', 'train')
	read_label('raw_data/train-labels-idx1-ubyte', 'train')

	#read_image('raw_data/t10k-images-idx3-ubyte', 'test')
	read_label('raw_data/t10k-labels-idx1-ubyte', 'test')
