# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:46:36 2020
注释：学习chenwei的文章中的自编码网络，不包含变分部分
@author: zkp
"""
import tensorflow.compat.v1 as tf

from numpy.random import RandomState
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os
import matplotlib
import cv2


tf.reset_default_graph()
matplotlib.use('qt5Agg')
###屏蔽waring信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
###地址
cwd = "I:\\zhangkunpeng\\bian_Cjin\\"

data_path=[]
for tf_i in range(0,3):
    data_path.append (cwd +'creatdata\\Generate_Images01053_lin_wid.tfrecords-%03d'%tf_i)


#获取文件名列表
data_files = tf.io.gfile.glob(data_path)
filename_queue = tf.train.string_input_producer(data_files,shuffle=True) 
reader = tf.TFRecordReader()

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.float32),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                       })

image0,label1 = features['img_raw'],features['label']
image1 = tf.decode_raw(image0, tf.uint8)
image1 = tf.cast(image1, tf.float32)
image2 = -tf.reshape(image1, [64,64,3])/255.0 + 1

batch_size = 100
capacity =3*batch_size
###顺序
#example_batch, label_batch=tf.train.batch([image2,label1], batch_size =batch_size,capacity = capacity)
###乱序
example_batch, label_batch= tf.train.shuffle_batch([image2,label1], batch_size =batch_size ,capacity = capacity,min_after_dequeue=20)

######
class_path_test= cwd +'creatdata\\Generate_Images01053_lin_wid.tfrecords-001'
data_files_test=tf.gfile.Glob(class_path_test)
filename_queue_test = tf.train.string_input_producer(data_files_test,shuffle=True) 
reader = tf.TFRecordReader()
_, serialized_example_test = reader.read(filename_queue_test)
#训练集合tfrecord读取文件内容
features_test = tf.parse_single_example(serialized_example_test,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.float32),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                       })

image_test0 , label_test1 = features_test['img_raw'],features_test['label']
image_test1 = tf.decode_raw(image_test0, tf.uint8)
image_test1 = tf.cast(image_test1, tf.float32) 
####类型转换tuxing 
image_test2 =- tf.reshape(image_test1, [64,64,3])/255.0+1
batch_size_test1 =40
capacity_test1 =3*batch_size_test1

example_batch_test1, label_batch_test1= tf.train.batch([image_test2,label_test1], batch_size =batch_size_test1 ,capacity = capacity_test1)


#入口
x_ = tf.placeholder(tf.float32, [None, 64,64,1],name='X_input')
x = tf.reshape(x_, [-1,64,64,1])

targets_ = tf.placeholder(tf.float32, (None, 64, 64, 1), name='Y_input')
targets_ = tf.reshape(targets_, [-1,64,64,1])

#====================================================================================
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  ##截断正态分布
    # initial = tf.random_normal(shape, stddev=0.1)  ##标准正态分布
    # initial = tf.variance_scaling_initializer(shape,)  ##
    # initial = tf.orthogonal_initializer(shape,-1,1)  ##正交矩阵的随机数
    # initial = tf.glorot_uniform_initializer(shape,)  ##Xavier uniform initializer，由一个均匀分布来初始化数据
    # initial = tf.random_uniform(shape, -0.1,0.1)  ##均匀分布
    # initial = tf.uniform_unit_scaling_initializer(shape,)  ##和均匀分布差不多
    return tf.Variable(initial, name="w")

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name="b")

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def max_pool_4d(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')

def mean_var_with_update():
    ema_apply_op = ema.apply([fc_mean, fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)

#初始化权重、偏置
def glorot_init(shape):
    return tf.random_normal(shape=shape,stddev=1./tf.sqrt(shape[0]/2.0))

#====================================================================================
###Encoder
Conv_num1 = 32
Conv_num2 = 32
Conv_num3 = 32
Conv_num4 = 64
Conv_num5 = 64
###图像处理0-1
def binary_activation(x):
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


x_01 = binary_activation(x-0.6)
# x_01 = x
with tf.variable_scope("conv1"):
    weights = weight_variable([3,3, 1, Conv_num1])
    biases = bias_variable([Conv_num1])
    conv11 = conv2d(x_01 , weights) + biases
    fc_mean, fc_var = tf.nn.moments(conv11,axes=[0,1,2],)
    scale = tf.Variable(tf.ones([Conv_num1]))
    shift = tf.Variable(tf.zeros([Conv_num1]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, var = mean_var_with_update()
    conv_gui1 = tf.nn.batch_normalization(conv11, fc_mean, fc_var, shift, scale, epsilon)
    conv13 = tf.nn.relu(conv_gui1)
    pool11 = conv13


with tf.variable_scope("conv2"):
    weights = weight_variable([3,3, Conv_num1, Conv_num2])
    biases = bias_variable([Conv_num2])
    conv21 = conv2d(pool11, weights) + biases
    
    fc_mean, fc_var = tf.nn.moments(conv21,axes=[0,1,2],)
    scale = tf.Variable(tf.ones([Conv_num2]))
    shift = tf.Variable(tf.zeros([Conv_num2]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, var = mean_var_with_update()
    conv_gui2 = tf.nn.batch_normalization(conv21, fc_mean, fc_var, shift, scale, epsilon)
    conv23 = tf.nn.relu(conv_gui2)
    pool22 = max_pool_2d(conv23)


with tf.variable_scope("conv3"):
    weights = weight_variable([3,3, Conv_num2, Conv_num3])
    biases = bias_variable([Conv_num3])
    conv31 = conv2d(pool22, weights) + biases
    
    fc_mean, fc_var = tf.nn.moments(conv31,axes=[0,1,2],)
    scale = tf.Variable(tf.ones([Conv_num3]))
    shift = tf.Variable(tf.zeros([Conv_num3]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, var = mean_var_with_update()
    conv_gui3 = tf.nn.batch_normalization(conv31, fc_mean, fc_var, shift, scale, epsilon)
    conv33 = tf.nn.relu(conv_gui3)
    pool33 = (conv33)

with tf.variable_scope("conv4"):
    weights = weight_variable([3,3, Conv_num3, Conv_num4])
    biases = bias_variable([Conv_num4])
    conv41 = conv2d(pool33, weights) + biases
    
    fc_mean, fc_var = tf.nn.moments(conv41,axes=[0,1,2],)
    scale = tf.Variable(tf.ones([Conv_num4]))
    shift = tf.Variable(tf.zeros([Conv_num4]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, var = mean_var_with_update()
    conv_gui4 = tf.nn.batch_normalization(conv41, fc_mean, fc_var, shift, scale, epsilon)
    conv43 = tf.nn.relu(conv_gui4)
    pool44 = conv43

with tf.variable_scope("conv5"):
    weights = weight_variable([3,3, Conv_num4, Conv_num5])
    biases = bias_variable([Conv_num5])
    conv51 = conv2d(pool44, weights) + biases
    
    fc_mean, fc_var = tf.nn.moments(conv51,axes=[0,1,2],)
    scale = tf.Variable(tf.ones([Conv_num5]))
    shift = tf.Variable(tf.zeros([Conv_num5]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, var = mean_var_with_update()
    conv_gui5 = tf.nn.batch_normalization(conv51, fc_mean, fc_var, shift, scale, epsilon)
    conv53 = tf.nn.relu(conv_gui5)
    # pool55 = max_pool_2d(conv53)

Fc_num1 = 32
Fc_num2 = 16
Fc_num3 = 1000

with tf.variable_scope("fc1"):
    W_fc1 = weight_variable([32*32 * Conv_num5, Fc_num1])
    poll1 = tf.reshape(conv53 , [-1,32 * 32* Conv_num5])
    b_fc1 = bias_variable([1,Fc_num1])
    h_fc11 = tf.matmul(poll1, W_fc1) + b_fc1
    fc_mean, fc_var = tf.nn.moments(h_fc11,axes=[0],)
    scale = tf.Variable(tf.ones([Fc_num1]))
    shift = tf.Variable(tf.zeros([Fc_num1]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    mean, var = mean_var_with_update()
    h_fc = tf.nn.batch_normalization(h_fc11, fc_mean, fc_var, shift, scale, epsilon)
    h_fc13 = tf.nn.relu(h_fc)


#权重
S_weights={'z_mean':tf.Variable(glorot_init([Fc_num1, Fc_num2])),
    'z_std':tf.Variable(glorot_init([Fc_num1, Fc_num2])),}
#偏置
S_biases={'z_mean':tf.Variable(glorot_init([Fc_num2])),
    'z_std':tf.Variable(glorot_init([Fc_num2])),}

##定义采样器
with tf.variable_scope("sampler"):
    W_Sam = weight_variable([Fc_num1, Fc_num2])
    b_Sam = bias_variable([1,Fc_num2])
    S_matmu = tf.matmul(h_fc13, W_Sam) + b_Sam
    S_mean = tf.matmul(h_fc13,S_weights['z_mean'])+S_biases['z_mean']
    S_std = tf.matmul(h_fc13,S_weights['z_std'])+S_biases['z_std']
    eps = tf.random_normal(tf.shape(S_std),dtype=tf.float32,mean=0,stddev=1.0,name='epsilon')
    S_mid=S_mean+tf.exp(S_std/2)*eps


decon_num1 = 32
decon_num2 = 32
decon_num3 = 1
#Decoder
with tf.variable_scope("fc3"):
    W_fc3 = weight_variable([Fc_num2, Fc_num3])
    b_fc3 = bias_variable([1,Fc_num3])
    h_fc31 = tf.matmul(S_mid, W_fc3) + b_fc3
    h_fc33 = tf.nn.relu(h_fc31)

with tf.variable_scope("fc4"):
    W_fc4 = weight_variable([Fc_num3, 32*32*decon_num1])
    b_fc4 = bias_variable([1,32*32*decon_num1])
    h_fc41 = tf.matmul(h_fc33, W_fc4) + b_fc4
    h_fc43 = tf.nn.relu(h_fc41)
    fc = tf.reshape(h_fc43, [-1, 32, 32, decon_num1])

with tf.variable_scope("deconv1"):  ###4*4*4
    de_conv1= tf.image.resize_nearest_neighbor(fc, (64,64))
    weights = weight_variable([3, 3, decon_num1, decon_num2])
    biases = bias_variable([decon_num2])
    de_conv41 = conv2d(de_conv1, weights) + biases
    de_conv43 = tf.nn.relu( de_conv41)

with tf.variable_scope("deconv2"):  ###4*4*4
    weights = weight_variable([3, 3, decon_num2, decon_num3])
    biases = bias_variable([decon_num3])
    de_conv51 = conv2d(de_conv43, weights) + biases
    h_conv53 = tf.nn.sigmoid(de_conv51)

logits_ = h_conv53
outputs_ = de_conv51
# outputs1_ = h_conv53

old_Pic =  binary_activation(targets_-0.6)

#定义损失函数
def vae_loss(x_reconstructed,x_true,z_mean,z_std):
    #重构损失
    encode_decode_loss=x_true*tf.log(1e-10+x_reconstructed)+(1-x_true)*tf.log(1e-10+1-x_reconstructed)
    encode_decode_loss=-tf.reduce_sum(encode_decode_loss,1)
    #KL损失
    kl_div_loss=1+z_std-tf.square(z_mean)-tf.exp(z_std)
    kl_div_loss=-0.01*tf.reduce_sum(kl_div_loss,1)
    return tf.reduce_mean(encode_decode_loss+kl_div_loss),encode_decode_loss,kl_div_loss


cost,ED_loss,KL_loss =vae_loss(logits_ ,old_Pic,S_mean,S_std)
###学习步调整，难点，需要学习原理
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = tf.placeholder(tf.float32)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,2000, 0.9, staircase=True)
####优化器
optimizer = tf.train.RMSPropOptimizer(learning_rate)
trainer = optimizer.minimize(cost,global_step)

###输出步数
prin_num =20
###总步数
train_num = prin_num*15
#用于绘图
xx=np.zeros(train_num)
yy=np.zeros(train_num)
###用于保存
saver = tf.train.Saver(max_to_keep=10)
#tf.add_to_collection("old_train", trainer)
#tf.add_to_collection("Pool11", pool11)
#tf.add_to_collection("full_fc2", h_fc2_drop)
#tf.add_to_collection("predict_out", logits_)
#plt.ion()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph) #第一个参数指定生成文件的目录。
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    ###在原来基础上继续训练的语句
    saver.restore(sess, cwd+"XTF_model\\VAE_D16_guiyi_First_neta-750")
    for i in range(1):
        (x_batch,Label)= sess.run([example_batch,label_batch])
        S_mean11 = sess.run(S_mean,feed_dict={x_:x_batch[:,:,:,[0]] ,})
        # # ss
        # x_batch.shape
        # xuexilv = 0.01
        train_new_pic0,train_old_pic = sess.run([outputs_, old_Pic],feed_dict={x_:x_batch[:,:,:,[0]] ,   targets_:x_batch[:,:,:,[0]] })
        
        
        # plt.subplot(1,10,1)
        # plt.imshow(train_old_pic[0])
        # plt.axis('off')
        
        # plt.subplot(1,10,2)
        # plt.imshow(train_new_pic0[0])
        # plt.axis('off')
        
        Mide_label_16 = sess.run(S_mid,feed_dict={x_:x_batch[:,:,:,[0]] ,})
        Mide_label_one =Mide_label_16[2]
        Mide_label_one = Mide_label_one.reshape([-1,16])
        for jj in range(10):
            Mide_label_one[0,1] = 0.1*jj
            for ii in range(10):
                Mide_label_one[0,2] = 0.1*ii 
                New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
                plt.subplot(10,10,jj*10+ii+1)
                plt.imshow(New_test[0])
                plt.axis('off')
        fig = matplotlib.pyplot.gcf()
        
        fig.savefig(cwd + 'AutoE_test_pic_xtf\\Vae12_varia%.5d.jpg'%i,dpi=100)
        plt.close()
        
        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,3] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,4] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae34_varia%.5d.jpg'%i,dpi=100)
        # plt.close()
        
        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,5] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,6] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae56_varia%.5d.jpg'%i,dpi=100)
        # plt.close()
        
        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,7] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,8] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae78_varia%.5d.jpg'%i,dpi=100)
        # plt.close()

        
        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,9] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,10] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae910_varia%.5d.jpg'%i,dpi=100)
        # plt.close()
        
        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,11] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,12] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae1112_varia%.5d.jpg'%i,dpi=100)
        # plt.close()

        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,13] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,14] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae1314_varia%.5d.jpg'%i,dpi=100)
        # plt.close()

        # Mide_label_one =Mide_label_16[0]
        # Mide_label_one = Mide_label_one.reshape([-1,16])
        # for jj in range(10):
        #     Mide_label_one[0,15] = 0.1*jj
        #     for ii in range(10):
        #         Mide_label_one[0,16] = 0.1*ii 
        #         New_test = sess.run(logits_,feed_dict ={S_mid:Mide_label_one})
        #         plt.subplot(10,10,jj*10+ii+1)
        #         plt.imshow(New_test[0])
        #         plt.axis('off')
        # fig = matplotlib.pyplot.gcf()
        # fig.savefig(cwd + 'AutoE_pic\\Vae1516_varia%.5d.jpg'%i,dpi=100)
        # plt.close()
        #     # Mide_label_one[0,0] = Mide_label_one[0,0]+0.1
        # ss
        # for jj in range(6):
            
        # if i % prin_num == 0:
        #      batch_cost,ED_loss1 , Kl_loss1= sess.run([cost,ED_loss,KL_loss],feed_dict={x_:x_batch[:,:,:,[0]] ,  targets_ :x_batch[:,:,:,[0]],starter_learning_rate : xuexilv})
        #      train_new_pic,train_old_pic = sess.run([outputs_, old_Pic],feed_dict={x_:x_batch[:,:,:,[0]] ,   targets_:x_batch[:,:,:,[0]] })
        #      print(batch_cost,max(ED_loss1[1]) , max(Kl_loss1))
        #      Mide_label_16 = sess.run(S_mid,feed_dict={x_:x_batch[:,:,:,[0]] ,})
             
        #      Mide_mean_16 = sess.run(S_mean,feed_dict={x_:x_batch[:,:,:,[0]] ,})
        #      Mide_Std_16 = sess.run(S_std,feed_dict={x_:x_batch[:,:,:,[0]] ,})
        #      ###训练集图片效果
        #      for pic_i in range(10):
        #          plt.subplot(4,13,pic_i+1)
        #          plt.imshow(train_old_pic[pic_i,:,:,0])
        #          plt.axis('off')
        #          plt.subplot(4,13,13+pic_i +1)
        #          plt.imshow(train_new_pic[pic_i,:,:,0])
        #          plt.axis('off')
        #          plt.subplot(4,13,13*2+pic_i +1)
        #          AA_03 = sess.run(binary_activation(train_new_pic[pic_i,:,:,0]-0.3))
        #          plt.imshow(AA_03)
        #          plt.axis('off')
        #          plt.subplot(4,13,13*3+pic_i +1)
        #          AA_05 = sess.run(binary_activation(train_new_pic[pic_i,:,:,0]-0.5))
        #          plt.imshow(AA_05)
        #          plt.axis('off')
                 
        #      x_test ,y_test_label = sess.run([example_batch_test1, label_batch_test1])
        #      batch_cost_test  = sess.run(cost,feed_dict={x_:x_test[:,:,:,[0]] ,  targets_ :x_test[:,:,:,[0]],starter_learning_rate : xuexilv})
        #      learning_lv = sess.run(learning_rate,feed_dict = {starter_learning_rate : xuexilv, })
        #      print("idx: {}/{} ".format(i+1, train_num),
        #              "Training loss: {:.6f}".format(batch_cost),
        #              "Ttesting loss: {:.6f}".format(batch_cost_test),
        #              "learning_rate.s: {:.6f}".format(learning_lv))
             
             
        #      test_new_pic,test_old_pic = sess.run([outputs_, old_Pic],feed_dict={x_:x_test[:,:,:,[0]] ,   targets_:x_test[:,:,:,[0]] })
        #      for test_pic_i in range(3):
        #          plt.subplot(4,13,10+test_pic_i+1)
        #          plt.imshow(test_old_pic[test_pic_i,:,:,0])
        #          plt.axis('off')
        #          plt.subplot(4,13,13+10+test_pic_i +1)
        #          plt.imshow(test_new_pic[test_pic_i,:,:,0])
        #          plt.axis('off')
        #          plt.subplot(4,13,13*2+10+test_pic_i +1)
        #          AA_test_03 = sess.run(binary_activation(test_new_pic[test_pic_i,:,:,0]-0.3))
        #          plt.imshow(AA_test_03)
        #          plt.axis('off')
        #          plt.subplot(4,13,13*3+10+test_pic_i +1)
        #          AA_test_05 = sess.run(binary_activation(test_new_pic[test_pic_i,:,:,0]-0.5))
        #          plt.imshow(AA_test_05)
        #          plt.axis('off')
             
        #      fig = matplotlib.pyplot.gcf()
        #      fig.savefig(cwd + 'AutoE_pic\\VAE_D16_guiyi_test_pic%.5d.jpg'%i,dpi=100)
        #      plt.close()
             
             # saver.save(sess,"I:\\zhangkunpeng\\bian_Cjin\\model\\VAE_D16_guiyi_test_neta", global_step=i)
            
    coord.request_stop()
    coord.join(threads)

#####图像处理：高于箱数0.5的当成1.0，其余的为0
#with tf.Session() as sess:
#    AA = sess.run(binary_activation(test_new_pic[test_pic_num,:,:,0]-0.1))
#    plt.imshow(AA)



