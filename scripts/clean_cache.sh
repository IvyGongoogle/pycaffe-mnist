##################################################
# Filename: clean_cache.sh
# Author: ChrisZZ
# E-mail: zchrissirhcz@163.com
# Created Time: 2017年01月04日 星期三 05时52分33秒
##################################################
#!/bin/bash
rm -f `find . -name '*.pyc'`
rm output/* -rf
#rm -f mnist_iter_5000.caffemodel
#rm -f mnist_iter_5000.solverstate
#mkdir -p output/train
#mkdir -p output/test
