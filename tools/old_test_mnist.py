#!/usr/bin/env python
# coding:utf-8

import _init_paths
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import os
import caffe
import google.protobuf.text_format
from utils.timer import Timer

def parse_args():
	"""parse input arguments"""
    parser = argparse.ArgumentParser(description='traffic_police demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--input', dest='input_dir', default='data/test')

    args = parser.parse_args()

    return args

def im2blob(im):
	"""
	将ndarray类型的图像im转化为blob(ndarray)
	"""

def im_classify(net, im):
	"""
	执行图像分类。（手写数字分类）
	@param im: opencv读取出来的图像,ndarray类型
	"""
	# 1.先把图像转化为blob
	# 2.然后网络做前馈
	# 3.取出网络前馈的输出
	# 4.返回结构

	blobs = {'data': None}
	blobs['data'] = _get_image_blob(im)
	net.blobs['data'].reshape(*(blobs['data'].shape))

def im_classify_2(net, im):
	"""
	执行图像分类。（手写数字分类）
	"""
	# 1.先把图像转化为blob
	# 2.然后网络做前馈
	# 3.取出网络前馈的输出
	# 4.返回结构

	# 先看第一种方式，来自jianh/test.py
	im_name = '1.png'  # 需要修改
	try:
		im = Image.open(im_name)
		im.load()
	except:
		print('fail to open image', im_name)
		return
	m = 28  # 用于resize的边长
	t1 = time.time()
	sized = np.float32(im.resize(m, m))
	if len(sized.shape)==2:
		sized3 = np.empty((m, m, 3), dtype=np.float32)
		for i in xrange(3):
			sized3[:, :, i] = sized[:, :]
		sized = sized3

	transformer = SimpleTransformer()
	in_ = transformer.preprocess(sized)
	# 或者:
	# my_transformer = caffe.io.Transformer({'data': .......})
	# caffe_img = caffe.io.load_image(fpath)
	# im_ = transformer.preprocess('data', caffe_img)

	t2 = time.time()
	t_pre = t2 - t1
	# 再看第二种方式

	t1 = time.time()
	# 整理输入数据的形状，设定data
	net.blobs['data'].reshape(1, 3, m, m)
	net.blobs['data'].data[...] = in_
	# 运行网络并找出最大值作为预测结果
	net.forward()
	t2 = time.time()
	t_forw = t2 - t1

	t1 = time.time()
	out = net.blobs['ip2'].data[0]

def demo(net, image_name):
	# 加载图片
	im_file = os.path.join(args.input_dir, image_name)
	im = cv2.imread(im_file)

	timer = Timer()
	timer.tic()
	# 执行分类
	pred_cls = im_classify(net, im)
	timer.toc()
	print('图像分类耗时{:3f}s, 图像本身为{:1d}, 分类结果为{:1d}').format(timer.total_time, gt_cls, pred_cls)

def do_mnist_test():
	# 执行测试，用户友好方式
	"""
	稍微想下，包括这几个步骤：
	指定网络prototxt
	指定权值文件caffemodel
	网络初始化
	在所有测试图片上遍历
		读取图片
		网络做inference
		取出网络的输出
		显示在图像上或者屏幕打印
	"""
	print 'Testing...'
	# caffe初始化
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)

	# 数据集初始化
	# imdb = datasets.imdb.imdb(phase='test')
	
	# 网络初始化
	prototxt = 'models/lenet/deploy.prototxt'
	caffemodel = 'output/train/mnist_iter_5000.caffemodel'
	if not os.path.isfile(prototxt):
		raise IOError(('{:s} not found.').format(caffemodel))
	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.').format(caffemodel))
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	print "\n\nLoaded network {:s}".format(caffemodel)

	# warmup on a dummy image. 是否可以省略？
	im = 128 * np.ones((28, 28, 1), dtype=np.uint8)
	for i in xrange(2):
		_, _ = im_classify(net, im)
	
	for filename in os.listdir(args.input_dir):
		demo(net, filename)

	print 'done Testing'
	
if __name__ == '__main__':
	args = parse_args()
	do_mnist_test()
