#!/bin/bash
cd /home/huangjiehui/Project/Mathematical/Spinal-Disease-Detection/Classification/Data-ForTest/Data
# 遍历当前目录下的所有文件夹
for dir in */; do
  # 获取当前文件夹的名字
  dir_name=`basename $dir`
  # 遍历当前文件夹下的所有文件
  for file in $dir*; do
    # 获取文件名和后缀
    base_name=`basename $file`
    suffix=${base_name##*.}
    base_name=${base_name%.*}
    # 对文件进行重命名
    mv $file $dir/${base_name}-$dir_name.$suffix
  done
done