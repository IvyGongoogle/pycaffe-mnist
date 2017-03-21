#!/usr/bin/env python
# coding:utf-8

from easydict import EasyDict as edict
import platform

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

## 训练阶段相关配置
__C.TRAIN = edict()

# 保存caffemodel的迭代间隔数
__C.TRAIN.SNAPSHOT_ITERS = 1000

# 训练迭代次数
__C.TRAIN.MAX_ITERS = 5000

# 每个mini-batch内图片数量
__C.TRAIN.BATCH_SIZE = 64

# LMDB
__C.TRAIN.TRAIN_LMDB = 'data/train_lmdb'
__C.TRAIN.VAL_LMDB = 'data/test_lmdb'

## 数据集相关配置
__C.DATASET = edict()
__C.DATASET.ID_START = 1

## meta信息
__C.META = edict()

sysstr = platform.system()
if sysstr == "Windows":
    __C.META.ROOT_DIR = 'd:/code/mnist'
elif sysstr == "Linux":
    __C.META.ROOT_DIR = '/home/chris/work/mnist'