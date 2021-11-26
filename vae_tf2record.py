# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:08:34 2021

@author: Administrator
"""
import glob
import os
import tensorflow as tf
from PIL import Image

# data_path='/data-input/pic2000'# windows里好像这个\\比较管用,与os相适配
data_path='I:\\zhangkunpeng\\bian_Cjin\\pic2000'# windows里好像这个\\比较管用,与os相适配
pic_list=glob.glob(os.path.join(data_path,'*.tiff'))



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



# 当下任务，生成tfrecord文件批次

record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for path in pic_list:
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




# raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# # Create a dictionary describing the features.
# image_feature_description = {
#     'height': tf.io.FixedLenFeature([], tf.int64),
#     'width': tf.io.FixedLenFeature([], tf.int64),
#     'image_raw': tf.io.FixedLenFeature([], tf.string),
# }

# def _parse_image_function(example_proto):
#   # Parse the input tf.Example proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, image_feature_description)

# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
# parsed_image_dataset