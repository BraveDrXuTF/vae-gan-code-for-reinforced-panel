# -*- coding: utf-8 -*-
"""
Created on Wed May 19

@author: xtf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from layer import MaxUnpooling2D,MaxPoolingWithArgmax2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import add
import IPython.display as display
import matplotlib  
matplotlib.use('Agg') # 这样可以不显示图窗
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import random
#计算数据集特征
def sampling(args):
    """Reparameterization trick by sampling 
        fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

##画图
def plot_results(models,batch):
    """Plots labels and MNIST digits as function 
        of 3-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    z3_list = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    for z3 in z3_list:
        filename = os.path.join('images', "digits_over_latent16w_ind{0}_z3is{1}.png".format(batch//500+1,z3))
        # display a 5x5 2D manifold of digits
        n = 5
        digit_size = 64
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]
        z3 = 0
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi, z3]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
        plt.figure(figsize=(15, 15))
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.close() #这句话保证图像不会重叠
##画图
def plot_resultsz2(models,epoch):
    """Plots labels and MNIST digits as function 
        of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models

    filename = os.path.join('images', "digits_over_latent16w_epoch{0}_resnet527.png".format(epoch))
    # display a 30x30 2D manifold of digits
    n = 25
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(12, 12))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.close() #这句话保证图像不会重叠
def test_plot(vae,a128batch,epoch):
    testresult=vae.predict(a128batch)
    testresult = np.squeeze(testresult)
    digit_size = 64
    row = 2
    col = 10
    figure = np.zeros((digit_size * 2 *row, digit_size * col)) # 4行10列
    for i in range(col):
        # import pdb
        # pdb.set_trace()
        figure[0:digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i]
        figure[1*digit_size:2*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i]
        figure[2*digit_size:3*digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i+10]
        figure[3*digit_size:4*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i+10]
    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('images/test_vaeres_epoch{0}.jpg'.format(epoch))
    plt.close()
# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)
def parse_imagestr2numpy(image_features):
    image1 = tf.io.decode_raw(image_features['image_raw'], tf.uint8)
    image1 = tf.cast(image1, tf.float32)
    image2 = -tf.reshape(image1, [64,64,3])/255.0 + 1
    image2 = image2[:,:,0]
    return image2

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def decode_resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2DTranspose(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

class MyPlotCallback(Callback):
    def __init__(self, models):
        self.models = models
    def on_train_batch_end(self, batch, logs=None):
        if batch%500==1:
            plot_results(models,batch)
class MyPlotCallbackz2(Callback):
    def __init__(self, models):
        self.models = models
    def on_epoch_end(self, epoch, logs=None):
        plot_resultsz2(models, epoch)
class MyPlotCallback_test(Callback):
    def __init__(self, vae, a128batch):
        self.vae = vae
        self.a128batch = a128batch
    def on_epoch_end(self, epoch, logs=None):
        test_plot(self.vae, self.a128batch, epoch)
class MyepochsaveCallback(Callback):
    def __init__(self, save_dir, vae):
        self.save_dir = save_dir
    def on_epoch_end(self, epoch, logs=None):
        vae.save_weights(os.path.join(self.save_dir,'modelepoch_{0}z2LC_resnet527.h5'.format(epoch)))
    def on_train_batch_end(self, batch, logs=None):
        if batch%500==1:
            vae.save_weights(os.path.join(self.save_dir,'modelbatch{0}z2LC_resnet527.h5'.format(batch)))
num_filters = 16
image_size = 64
# tfrecord_list=glob.glob('chapter8-vae/data/*.tfrecords') #因为vs code是在Advanced文件夹打开的
tfrecord_list=glob.glob('data/*.tfrecords')
# tfrecord_list = ['data/images_batch0.tfrecords','data/images_batch1.tfrecords']
image_list_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)
parsed_image_dataset = image_list_dataset.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_image_function),
    cycle_length=4)

# parsed_image_dataset = tf.data.TFRecordDataset(tfrecord_list)


batch_size = 128
batch_num = 1302
# 原來dataset可以這樣做
parsed_image_dataset = parsed_image_dataset.map(parse_imagestr2numpy)
parsed_image_dataset = parsed_image_dataset.batch(128)
a128batch=parsed_image_dataset.take(1)
a128batch_list = list(a128batch.as_numpy_iterator())
# train_data = parsed_image_dataset.take(1000)
# test_data = parsed_image_dataset.skip(1000)
# network parameters
input_shape = (image_size, image_size, 1)
kernel_size = 3
filters = 16
latent_dim = 2 #曾经为3改回2，因为第三个维度不管用，现在使用后缀为z2的callback
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# 残差网络版本
num_res_blocks = 3 # depth是去计算残差块的数目的,这里的残差结构默认depth=20(见Advanced Keras书籍代码)
for stack in range(3):
    for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        y = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        strides=strides)
        y = resnet_layer(inputs=y,
                            num_filters=num_filters,
                            activation=None)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut connection to match
            # changed dims
            x = resnet_layer(inputs=x,
                            num_filters=num_filters,
                            kernel_size=1,
                            strides=strides,
                            activation=None,
                            batch_normalization=False)
        x = add([x, y])
        x = Activation('relu')(x)
    num_filters *= 2
x,mask = MaxPoolingWithArgmax2D(pool_size=(2,2),strides=(2,2))(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary 
# with the TensorFlow backend
z = Lambda(sampling,
           output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])

# instantiate encoder model 这里应该额外输出mask
encoder = Model(inputs, [z_mean, z_log_var, z, mask], name='encoder')
encoder.summary()
plot_model(encoder,
           to_file='vae_cnn_encoder.png', 
           show_shapes=True)

# build decoder model
# 除了传递z，还要传递mask，这需要输出mask
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
mask_inputs = Input(shape=shape[1:],name='unpool_mask') #shape[1:]而不是shape，这很重要，因为shape的值是(None,8,8,64),tensorflow定义网络层张量的时候不用指定batch_size
x = Dense(shape[1] * shape[2] * shape[3],
          activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = MaxUnpooling2D(size=(2,2))([x,mask_inputs]) # 逆池化，对应前面的平均池化操作
for stack in reversed(range(3)):
    num_filters //= 2 #按照对称性，应该把num_filter放在这里
    for res_block in reversed(range(num_res_blocks)):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # upsample
        y = decode_resnet_layer(inputs=x,
                        num_filters=num_filters,
                        strides=strides)
        y = decode_resnet_layer(inputs=y,
                            num_filters=num_filters,
                            activation=None)
        # if stack > 0 and res_block == 0:  # first layer but not first stack
        #     # linear projection residual shortcut connection to match
        #     # changed dims
        x = decode_resnet_layer(inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False)
        x = add([y,x])
        x = Activation('relu')(x)
    

outputs = Conv2DTranspose(filters=1,
                kernel_size=1,
                activation='sigmoid',
                padding='same',
                name='decoder_output')(x)

# instantiate decoder model
decoder = Model([latent_inputs,mask_inputs], outputs, name='decoder')
decoder.summary()
plot_model(decoder,
           to_file='vae_cnn_decoder.png', 
           show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2:]) #取2，3不要写成2:3
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load tf model trained weights"
    parser.add_argument("--weights",help=help_)
    help_ = "Use binary cross entropy instead of mse (default)"
    parser.add_argument("--bce", help=help_, action='store_true')
    parser.add_argument("--cp2", help=help_)
    parser.add_argument("--cpz2", help=help_)
    parser.add_argument("--cpz2LC", help=help_)
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    # data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.bce:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))
    else:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adagrad')
    vae.summary()
    # plot_model(vae, to_file='C:\\Users\\Administrator\\Desktop\\Advanced-Deep-Learning-with-Keras-master\\chapter8-vae\\vaez2.png', show_shapes=True)
    
    save_dir = "vae_cnn_weights"
    # Checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(save_dir,
    #                                              save_weights_only=True,
    #                                              verbose=1)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.weights:
        filepath = os.path.join(save_dir, args.weights)
        vae.load_weights(filepath)
        # # 查看前128张图片的训练效果
        testresult=vae.predict(a128batch)
        testresult = np.squeeze(testresult)
        digit_size = 64
        row = 2
        col = 10
        figure = np.zeros((digit_size * 2*row, digit_size * col))
        for i in range(col):
            # import pdb
            # pdb.set_trace()
            figure[0:digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i]
            figure[1*digit_size:2*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i]
            figure[2*digit_size:3*digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i+10]
            figure[3*digit_size:4*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i+10]
        plt.figure(figsize=(15, 15))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig('try.jpg')
        plt.close()
            # for j in range(row):
            #     figure[j * digit_size: (j + 1) * digit_size,
            #        i * digit_size: (i + 1) * digit_size] = digit
    else:
        # train the autoencoder
        if args.cp2:
            filepath = os.path.join(save_dir, args.cp)
            vae.load_weights(filepath)
            vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallback(models),MyepochsaveCallback(save_dir,vae)])
            filepath = 'vae_jiajin_16w.h5'
            vae.save_weights(filepath)
        
        elif args.cpz2:
            filepath = os.path.join(save_dir, args.cpz2)
            vae.load_weights(filepath)
            vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallback_test(vae,a128batch),MyepochsaveCallback(save_dir,vae)])
            filepath = 'vae_jiajin_16wLC_resnet527.h5'
            vae.save_weights(filepath)
        elif args.cpz2LC:
            filepath = os.path.join(save_dir, args.cpz2LC)
            vae.load_weights(filepath)
            vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])
            filepath = 'vae_jiajin_16wLC_resnet527.h5'
            vae.save_weights(filepath)
        elif args.train:
            vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallback_test(vae,a128batch),MyepochsaveCallback(save_dir,vae)])
            filepath = 'vae_jiajin_16wLC_resnet527.h5'
            vae.save_weights(filepath)
