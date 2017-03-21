#!/usr/bin/env python
# coding:utf-8

import _init_paths
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import os.path
import datasets.imdb
import caffe
import platform
import google.protobuf.text_format
from config import cfg

class SolverWrapper(object):
	
	def __init__(self, solver_prototxt, output_dir, imdb, pretrained_model=None):
		self.output_dir = output_dir
		self.solver = caffe.SGDSolver(solver_prototxt)
		self.solver_param = caffe_pb2.SolverParameter()
		self.imdb = imdb
		with open(solver_prototxt, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)  # TODO：是否有必要？
		self.solver.net.layers[0].prepare_imdb(imdb)   # TODO: 有待实现。设定网络第一层的输入blob
	
	def snapshot(self):
		net = self.solver.net
		filename = self.solver_param.snapshot_prefix + '_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
		filename = ''.join(os.path.join(self.output_dir, filename))
		filename = unicode.encode(filename)   # 去掉恼人的前置u
		net.save(filename)
		#net.save(str(filename))
		print 'Wrote snapshot to: {:s}'.format(filename)
		return filename  # 有必要返回一个值吗？有，尤其是中途停掉的情况
	
	def train_model(self):
		"""
		@param max_iters:训练最大的迭代次数
		"""
		last_snapshot_iter = -1
		model_paths = []
		while self.solver.iter < cfg.TRAIN.MAX_ITERS:
			self.solver.step(1)
			if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				model_paths.append(self.snapshot())
				last_snapshot_iter = self.solver.iter
			predicted_labels = self.solver.net.blobs['ip2'].data[0].flatten()
			# print 'input blob of loss layer is:{}'.format(predicted_labels)
		if last_snapshot_iter != self.solver.iter:
			model_paths.append(self.snapshot())
		return model_paths

def do_mnist_train():
	"""
	稍微想一下，需要做的几个部件：
	caffe初始化
	定义网络的prototxt，并用它初始化caffe
	网络输入层的数据处理，需要从图像转化为imdb
	开始训练，可能需要自定义一些层
	一些工具函数需要自行编写呢
	"""
	
	# caffe的初始化
	caffe.set_mode_gpu()
	caffe.set_device(0)
	
	# 数据集初始化
	imdb = datasets.imdb.imdb(phase='train')
	
	# 网络初始化
	solver_prototxt = 'code/model/prototxt/lenet/lenet_solver.prototxt'
	output_dir = 'data/output/train'
	if os.path.exists(output_dir) is False:
		os.makedirs(output_dir)
	solver = SolverWrapper(solver_prototxt, output_dir, imdb)

	# 执行训练
	print 'Solving...'
	model_paths = solver.train_model()
	print 'done Solving'
	print model_paths
	
if __name__ == '__main__':
	root_dir = '/home/chris/work/mnist'
	os.chdir(root_dir)
	do_mnist_train()
