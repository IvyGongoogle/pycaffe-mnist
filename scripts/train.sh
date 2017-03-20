#!/bin/bash
##################################################
# Filename: train.sh
# Author: ChrisZZ
# E-mail: zchrissirhcz@163.com
# Created Time: 2017年01月04日 星期三 15时08分53秒
##################################################

LOG='logs/train.log'
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_mnist.py
