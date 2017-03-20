#!/usr/bin/env python
# coding: utf-8
# This scripts downloads the mnist data and unzips it.

import wget
import os
import gzip
import shutil

def main():
	root_dir = 'd:/code/mnist'
	os.chdir(os.path.join(root_dir, 'data'))
	if os.path.exists('raw_data') is False:
		os.mkdir('raw_data')
	
	os.chdir('raw_data')

	prefix = 'http://yann.lecun.com/exdb/mnist/{:s}.gz'
	items = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
	for item in items:
		url = prefix.format(item)
		# filename = wget.download(url)
		filename = item+'.gz'
		g_file = gzip.GzipFile(filename)
		open(item, 'w+').write(g_file.read())
		g_file.close()
		
if __name__ == '__main__':
	main()