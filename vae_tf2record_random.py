# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:08:30 2021

@author: Administrator
"""
import glob
import os
import tensorflow as tf
from PIL import Image
import random
# data_path='/data-input/pic2000'# windows里好像这个\\比较管用,与os相适配
data_path='I:\\zhangkunpeng\\bian_Cjin\\Generate_Images'# windows里好像这个\\比较管用,与os相适配
pic_list=glob.glob(os.path.join(data_path,'*.tiff'))
random.shuffle(pic_list) #彻底打乱顺序


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


batch_size = 128
# 当下任务，生成tfrecord批次文件，到时候每个batch直接一个文件放进去
batch_num = len(pic_list)//batch_size # 这里比较巧正好整除了，不用再加1
# 循环生成batch文件
for i in range(batch_num):
    record_file = 'data_random/images_batch{}.tfrecords'.format(i)
    with tf.io.TFRecordWriter(record_file) as writer:
        for path in pic_list[i*batch_size:(i+1)*batch_size]:
    #         image_string = open(path, 'rb').read() 这直接读好像对tiff格式的图片会出错，InvalidArgumentError: Input to reshape is a tensor with 2360 values, but the requested shape has 12288 [Op:Reshape]
            img=Image.open(path,'r')
            image_string = img.tobytes()
            image_shape = img.size # (64, 64)
            feature = {
              'height': _int64_feature(image_shape[0]),
              'width': _int64_feature(image_shape[1]),
              'image_raw': _bytes_feature(image_string),}
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
        writer.close()