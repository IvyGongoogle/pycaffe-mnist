# coding:utf-8
# description: 先前用py-lmdb生成了mnist的lmdb数据
#              现在尝试用py-lmdb进行读取，测试的方法是把数据和label读取出来

import _init_paths
import lmdb
from config import cfg
from caffe.proto import caffe_pb2
import os
import caffe
import numpy as np
import cv2

def main():
    """
    从头遍历所有图片
    """
    db = lmdb.open('train_lmdb')
    txn = db.begin()
    cursor = txn.cursor()
    datum = caffe_pb2.Datum()

    cnt = 0
    for key, value in cursor:
        datum.ParseFromString(value)

        label = datum.label
        data = caffe.io.datum_to_array(datum)

        # CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))
        if cnt == 2:
            cv2.imwrite('test.png', image)
            print('{},{}'.format(key, label))
            break
        cnt += 1
        print(cnt)

def main2():
    """
    读取指定位置的某一张图
    """
    db = lmdb.open('train_lmdb')
    txn = db.begin()
    cursor = txn.cursor()
    datum = caffe_pb2.Datum()

    print(len(cursor))
    
        

if __name__ == '__main__':
    root_dir = cfg.META.ROOT_DIR
    data_dir = os.path.join(root_dir, 'data')
    os.chdir(data_dir)
    # main()
    main2()
    print('Done.')