#!/usr/bin/env python
#coding: utf-8

"""添加pycaffe目录和lib目录到import搜索路径中"""

import os.path as osp
import sys
import platform

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

sysstr = platform.system()
if sysstr == "Windows":
    # 添加pycaffe目录到PYTHONPATH
    caffe_dir = osp.join('D:/lib/caffe-rfcn')
    pycaffe_dir = osp.join(caffe_dir, 'Build/x64/Release/pycaffe')

    # Add lib to PYTHONPATH
    lib_path = osp.join(this_dir, '..', 'lib')
    
elif sysstr == "Linux":
    # 添加pycaffe目录到PYTHONPATH
    caffe_dir = osp.join('/home/chris/work/caffe-fast-rcnn')
    pycaffe_dir = osp.join(caffe_dir, 'python')

    # Add lib to PYTHONPATH
    lib_path = osp.join(this_dir, '..', 'lib')

sys.path.insert(0, pycaffe_dir)
sys.path.insert(0, lib_path)
