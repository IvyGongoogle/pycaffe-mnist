#!/usr/bin/env python
# coding:utf-8

import _init_paths
import os
import glob
import random
import caffe
from PIL import Image
from config import cfg
import numpy as np
from caffe.proto import caffe_pb2
import lmdb
import time
import shutil

def make_datum_mnist(im, label):
    datum = caffe_pb2.Datum()
    datum.height, datum.width, datum.channels = im.shape
    if im.dtype == np.uint8:
        datum.data = im.tostring()
        # print('hoh')
    else:
        datum.float_data.extend(im.flat)
    datum.label = label
    return datum

def create_batch_lmdb(db, batch):
    """
    write(key,value) to db
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            for k, v in batch.items():
                txn.put(k, v)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            cur_limit = db.info()['map_size']
            new_limit = cur_limit * 2
            print '>>> Doubling LMDB map size to %sMB ...' % (new_limit>>20,)
            db.set_mapsize(new_limit)

def create_lmdb(lmdb_dir, label_filename):
    

    if os.path.exists(lmdb_dir):
        # os.rmdir(train_lmdb_dir)
        shutil.rmtree(lmdb_dir)

    label_id = 0
    label_mapper = {}
    with open('labels.txt') as f:
        for line in f:
            label_str = line.strip()
            label_mapper[label_str] = label_id
            label_id += 1

    ext = '.png'
    cnt = 0
    batch = {}

    db = lmdb.open(lmdb_dir, map_size=1048576)  # 1024*1024B = 1024K = 1M

    with open(label_filename) as f:
        for line in f.readlines():
            im_name, label_str = line.strip().split(' ')
            im_path = im_name + ext
            im = np.array(Image.open(im_path)) # dtype is uint8
            im = im[:,:,np.newaxis]
            # print('!! im.shape is', im.shape)
            label = label_mapper[label_str]
            
            datum = make_datum_mnist(im, label)
            keystr = '{:0>8d}'.format(cnt)
            value = datum.SerializeToString()
            batch[keystr] = value

            cnt += 1
            if cnt%1000==0:
                create_batch_lmdb(db, batch)
                batch = {}
        
        if cnt % 1000 != 0:
            create_batch_lmdb(db, batch)

    db.close()    

if __name__ == '__main__':
    root_dir = cfg.META.ROOT_DIR
    data_dir = os.path.join(root_dir, 'data')
    os.chdir(data_dir)

    start = time.time()
    create_lmdb(lmdb_dir = 'train_lmdb', label_filename='train_label.txt')
    create_lmdb(lmdb_dir = 'test_lmdb', label_filename='test_label.txt')
    print('总共耗时{:f}秒'.format(time.time()-start))
