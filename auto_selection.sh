#!/bin/bash
echo -n -e "say:"
read
echo -n -e "ehfe:"
read
# 第一个关键技术 执行命令输出到变量 当然可能没有用处 在本脚本中采用loss.txt保存每个python脚本执行的结果
COUNT=`python vaer_gan_z16.py`
echo $COUNT
