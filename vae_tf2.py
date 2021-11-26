# -*- coding: utf-8 -*-
"""
 9 17  2021

@author: xtf
看原版本需要重新解压usb
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
import IPython.display as display
import matplotlib  
# matplotlib.use('Agg') # 这样可以不显示图窗
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

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
        digit_size = 128
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

    filename = os.path.join('images', "digits_over_latent16w_epoch_weighted{0}.png".format(epoch))
    # display a 30x30 2D manifold of digits
    n = 25
    digit_size = 128
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
    plt.close() # 这句话保证图像不会重叠
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
    image2 = -tf.reshape(image1, [128,128,3])/255.0 + 1
    image2 = image2[:,:,0]
    return image2
# def testFunc():
    
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
class MyepochsaveCallback(Callback):
    def __init__(self, save_dir, vae):
        self.save_dir = save_dir
    def on_epoch_end(self, epoch, logs=None):
        vae.save_weights(os.path.join(self.save_dir,'modelepoch_{0}z2_weighted.h5'.format(epoch)))

image_size = 128
tfrecord_list=glob.glob('data_random_new/*.tfrecords')
# tfrecord_list = ['data/images_batch0.tfrecords','data/images_batch1.tfrecords']
image_list_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)
parsed_image_dataset = image_list_dataset.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_image_function),
    cycle_length=4)

# parsed_image_dataset = tf.data.TFRecordDataset(tfrecord_list)


batch_size = 4
batch_num = 1124
# 原來dataset可以這樣做
parsed_image_dataset = parsed_image_dataset.map(parse_imagestr2numpy)
parsed_image_dataset = parsed_image_dataset.batch(batch_size)
# train_data = parsed_image_dataset.take(1000)
# test_data = parsed_image_dataset.skip(1000)
# network parameters
input_shape = (image_size, image_size, 1)

kernel_size = 3
filters = 16
latent_dim = 2 # 曾经为3改回2，因为第三个维度不管用，现在使用后缀为z2的callback
epochs = 100







def reconstructed_loss(real_img,reconstructed_img):
    reconstructed_img = K.flatten(reconstructed_img)
    real_img = K.flatten(real_img) # 显示shape为(None,)
    difference = reconstructed_img-real_img
    weighted_img_difference = tf.multiply(K.square(difference),0.9+2*real_img)
    return image_size*image_size*tf.reduce_mean(weighted_img_difference,axis=-1)




# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
#3层卷积
for i in range(3):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)
print('shape 是 {}'.format(shape))
# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(latent_dim, activation=None)(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary 
# with the TensorFlow backend
z = Lambda(sampling,
           output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()



# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3],
          activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(3):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load tf model trained weights"
    parser.add_argument("--weights",help=help_)
    help_ = "Use binary cross entropy instead of mse (default)"
    parser.add_argument("--bce", help=help_, action='store_true')
    parser.add_argument("--cp2", help=help_)
    parser.add_argument("--cpz2", help=help_)
    parser.add_argument("--train", action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    # data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.bce:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))
    else:
        reconstruction_loss = reconstructed_loss(inputs,outputs)

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    
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
        a128batch=parsed_image_dataset.take(1)
        a128batch_list = list(a128batch.as_numpy_iterator())
        testresult=vae.predict(a128batch)
        testresult = np.squeeze(testresult)
        digit_size = 128
        row = 2
        col = 2
        figure = np.zeros((digit_size * row, digit_size * col))
        for i in range(col):
            # import pdb
            # pdb.set_trace()
            figure[0:digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i+col]
            figure[1*digit_size:2*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i+col]
        plt.figure(figsize=(15, 15))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig('try.jpg')
        plt.show()
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
            vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])
            filepath = 'vae_jiajin_16w.h5'
            vae.save_weights(filepath)

        elif args.train:
            vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])
            filepath = 'vae_jiajin_16w.h5'
            vae.save_weights(filepath)
            
        vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])