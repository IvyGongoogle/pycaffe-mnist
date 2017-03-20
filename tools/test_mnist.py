#!/usr/bin/env python
# coding:utf-8

import _init_paths
import caffe
import numpy as np
import os

## 先建立网络
root = '/home/chris/work/mnist'   #根目录
deploy=os.path.join(root, 'models/lenet', 'lenet_test.prototxt')    #deploy文件

caffe_model = os.path.join(root, 'output/train', 'mnist_iter_5000.caffemodel')   #训练好的 caffemodel

labels_filename = os.path.join(root, 'tools/labels.txt')  #类别名称文件，将数字标签转换回类别名称

print("deploy:", deploy)
print("caffe_model", caffe_model)
net = caffe.Net(deploy, caffe_model, caffe.TEST)   #加载model和network

## 网络输入设定：图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,1)变为(1,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间

# transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR
# 因为是单通道图像，所以不用交换通道


## 逐一处理各图片
input_dir = os.path.join(root, 'data/test')
#for filename in os.listdir(input_dir):
#fpath = os.path.join(input_dir, filename)
fpath = os.path.join(input_dir, '7409.png')
im=caffe.io.load_image(fpath, color=False)  # 加载图片.此处踩坑。color默认值为True，会把灰度图（单通道）存储为彩图（3通道），实际上mnist数据是单通道的。cv2.imread()默认也会将其转为3通道，很坑
print("im.shape is", im.shape)

net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中

# 网络做inference
out = net.forward()
print('out:')
print(out)

labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件

## 输出网络输出结果
# 方式1：适合于输出任何一层的东西
# prob= net.blobs['Softmax1'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印

# 方式2：仅输出整个网络中的“尽头层”的top
prob = out['Softmax1'].flatten()

## 将输出结果和标定数据对应上进行输出
print prob
order=prob.argsort()[-1]  #将概率值排序，取出最大值所在的序号 
print 'the class is:',labels[order]   #将该序号转换成对应的类别名称，并打印
