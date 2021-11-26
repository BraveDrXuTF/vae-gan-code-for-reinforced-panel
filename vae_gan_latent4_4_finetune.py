# -*- coding: utf-8 -*-
"""
Created on Wed May 30

@author: xtf
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from json import encoder
from tokenize import generate_tokens
from tensorflow import keras

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import IPython.display as display
import matplotlib
from tensorflow.python import training
from tensorflow.python.ops.nn_impl import weighted_cross_entropy_with_logits
from tensorflow.python.ops.variables import trainable_variables  
matplotlib.use('Agg') # 这样可以不显示图窗
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
        z (tensor): sampled latent map
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim, dim, 1))
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

    filename = os.path.join('images', "digits_over_latent16w_epoch{0}LC40907.png".format(epoch))
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
    image2 = image2[:,:,0] #把这一行去掉就可以实现三通道图片的输入
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
        vae.save_weights(os.path.join(self.save_dir,'modelepoch_{0}z2LC40907.h5'.format(epoch)))
    def on_train_batch_end(self, batch, logs=None):
        if batch%500==1:
            vae.save_weights(os.path.join(self.save_dir,'modelbatch{0}z2LC40907.h5'.format(batch)))
image_size = 64
tfrecord_list=glob.glob('data_random/*.tfrecords')
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
# train_data = parsed_image_dataset.take(1000)
# test_data = parsed_image_dataset.skip(1000)
# network parameters
input_shape = (image_size, image_size, 1)

kernel_size = 3
filters = 16
latent_dim = 4 #tensor width,这个变量是用来生成gen_z的
epochs = 100

# VAE model = encoder + decoder
# build encoder model
def build_encoder(filters=32):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # #三层卷积，前两层是strides=2的，后一层是stride=4,因此可以实现4*4的潜向量map
    # for i in range(2):
    #     x = Conv2D(filters=filters,
    #             kernel_size=kernel_size,
    #             activation='relu',
    #             strides=2,
    #             padding='same')(x)
    #     filters*=2

    # x = Conv2D(filters=filters,
    #         kernel_size=kernel_size,
    #         activation='relu',
    #         strides=2,
    #         padding='same')(x)    

    #第二种想法，四层卷积
    for i in range(4):
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
        filters*=2

    # shape info needed to build decoder model 
    # 4*4 map
    shape = K.int_shape(x)
    print('shape是{}'.format(shape))
    x = Conv2D(filters=1,
                kernel_size=kernel_size,
                activation=None,
                strides=1,
                padding='same')(x)

    z_mean = Conv2D(filters=1,
                kernel_size=kernel_size,
                activation=None,
                strides=1,
                padding='same', name='z_mean')(x)
    z_log_var = Conv2D(filters=1,
                kernel_size=kernel_size,
                activation=None,
                strides=1,
                padding='same', name='z_log_var')(x)
    # z_mean = Dense(latent_dim, name='z_mean')(x)
    # z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary 
    # with the TensorFlow backend
    z = Lambda(sampling,
            output_shape=(shape[1],shape[2],1), 
            name='z')([z_mean, z_log_var])

    # instantiate encoder model
    return Model(inputs, [z_mean, z_log_var, z], name='encoder')
    

def build_decoder(filters=256):
    # build decoder model
    latent_inputs = Input(shape=(4,4,1), name='z_sampling')

    x = latent_inputs

    for i in range(4):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2
    # 经过debug ,确认为64*64*1
    outputs = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)

    # instantiate decoder model
    return Model(latent_inputs, outputs, name='decoder_or_generator')



def build_discriminator(filters=16):
    inputs = Input(shape=(64,64,1), name='dis_input')
    x = inputs
    #3层卷积
    for i in range(3):
        filters *= 2
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1,activation='relu')(x)
    return Model(inputs, x, name='discriminator')

# 是否需要把strategy加上
class VAE_GAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        encoder
    ):
        super(VAE_GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
    def compile(
        self,
        encoder_optimizer,
        generator_optimizer,
        discriminator_optimizer,
        reconstructed_loss,
        kl_loss,
        discriminator_loss,
        gen_about_discriminator_loss
    ):
        super(VAE_GAN, self).compile()
        self.encoder_optimizer = encoder_optimizer
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer
        self.reconstructed_loss = reconstructed_loss
        self.kl_loss = kl_loss
        self.discriminator_loss = discriminator_loss
        self.gen_about_discriminator_loss = gen_about_discriminator_loss
    def train_step(self, one_batch_data):
        # input_image, target = one_batch_data
        real_img = one_batch_data
        gen_z = tf.random.normal([batch_size,latent_dim,latent_dim])
        with tf.GradientTape(persistent=True) as tape:

            # photo to monet back to photo
            z_mean, z_log_var, z = self.encoder(real_img, training=True)
            reconstructed_img = self.generator(z, training=True)
            generated_img = self.generator(gen_z, training=True)

            real_output = self.discriminator(real_img, training=True)
            reconstructed_output = self.discriminator(reconstructed_img, training=True)
            generated_output = self.discriminator(generated_img, training=True)
            
            
            reconstruction_loss = self.reconstructed_loss(real_img,reconstructed_img)
            kl_loss = self.kl_loss(z_mean,z_log_var)
            discriminator_loss = self.discriminator_loss(real_output,reconstructed_output,generated_output)

            vae_loss = reconstruction_loss + kl_loss
            gen_about_discriminator_loss = self.gen_about_discriminator_loss(reconstructed_output,generated_output)
        # Calculate the gradients for generator and discriminator
        encoder_gradients = tape.gradient(vae_loss,self.encoder.trainable_variables)

        generator_gradients1 = tape.gradient(reconstruction_loss, self.generator.trainable_variables)

        generator_gradients2 = tape.gradient(gen_about_discriminator_loss, self.generator.trainable_variables)

        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)


        self.encoder_optimizer.apply_gradients(zip(encoder_gradients,self.encoder.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(generator_gradients1,self.generator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(generator_gradients2,self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

        
        return {
            "vae_loss": vae_loss,
            "kl_loss": kl_loss,
            "reconstruction_loss": reconstruction_loss,
            "discriminator_loss": discriminator_loss
        }
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed_img = self.generator(z)
        return reconstructed_img

# with strategy.scope():
#     generator=Generator()
#     disc=discriminator_model()
# with strategy.scope():
#     cycle_gan_model = CycleGan(
#         generator,disc
#     )

#     cycle_gan_model.compile(
#         gen_optimizer = generator_optimizer,
#         disc_optimizer = discriminator_optimizer,
#         gen_loss_function = gen_loss,
#         disc_loss_function = disc_loss1,
#     )

#         encoder_optimizer,
#         generator_optimizer,
#         discriminator_optimizer,
#         reconstructed_loss,
#         kl_loss,
#         discriminator_loss
# z_mean, z_log_var, z = self.encoder(real_img, training=True)
# reconstructed_img = self.generator(z, training=True)
# generated_img = self.generator(gen_z, training=True)
# reconstruction_loss = self.reconstructed_loss(real_img,reconstructed_img)
# kl_loss = self.kl_loss(z_mean,z_log_var)
# discriminator_loss = self.discriminator_loss(real_img,reconstructed_img,generated_img)
encoder_optimizer = tf.keras.optimizers.RMSprop()
generator_optimizer = tf.keras.optimizers.RMSprop()
discriminator_optimizer = tf.keras.optimizers.RMSprop()
# 这个采用weighted的想法非常新颖
def reconstructed_loss(real_img,reconstructed_img):
    reconstructed_img = K.flatten(reconstructed_img)
    real_img = K.flatten(real_img) # 显示shape为(None,)
    difference = reconstructed_img-real_img
    weighted_img_difference = tf.multiply(K.square(difference),0.9+2*real_img)
    return image_size*image_size*tf.reduce_mean(weighted_img_difference,axis=-1)
# 这是native mse版本，当然，这里没有求平均
# def reconstructed_loss(real_img,reconstructed_img):
#     return image_size*image_size*mse(K.flatten(real_img), K.flatten(reconstructed_img))
# 我在维度方面仍然存在疑惑
def kl_loss(z_mean,z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss
# discriminator_loss使用的不是图片，而是关于图片的那些输出
def discriminator_loss(real_output,reconstructed_output,generated_output):
    loss1 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real_output),real_output)
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(reconstructed_output),reconstructed_output)
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated_output),generated_output)
    return loss1 + loss2 + loss3

def gen_about_discriminator_loss(reconstructed_output,generated_output):
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(reconstructed_output),reconstructed_output)
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated_output),generated_output) 
    return loss2 + loss3   
    # if args.bce:
    #     reconstruction_loss = binary_crossentropy(K.flatten(inputs),
    #                                               K.flatten(outputs))
    # else:
    #     reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))

    # reconstruction_loss *= image_size * image_size
    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)



# 实例化对象
encoder_instance = build_encoder()
generator_instance = build_decoder()
discriminator_instance = build_discriminator()

vae_gan_instance = VAE_GAN(generator_instance,discriminator_instance,encoder_instance)
# 编译vae_gan
vae_gan_instance.compile(encoder_optimizer,generator_optimizer,discriminator_optimizer,reconstructed_loss,kl_loss,discriminator_loss,gen_about_discriminator_loss)

def test_plot(model,a128batch,epoch):
    testresult=model.predict(a128batch)
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
    plt.savefig('images/vaegan4_epoch{0}.jpg'.format(epoch))
    plt.close()

class MyPlotCallback_test(Callback):
    def __init__(self, model, a128batch):
        self.model = model
        self.a128batch = a128batch
    def on_epoch_end(self, epoch, logs=None):
        test_plot(self.model, self.a128batch, epoch)
        

if __name__ == '__main__':
    a128batch=parsed_image_dataset.take(1)
    a128batch_list = list(a128batch.as_numpy_iterator())
    # parser = argparse.ArgumentParser()
    # help_ = "Load tf model trained weights"
    # parser.add_argument("--weights",help=help_)
    # help_ = "Use binary cross entropy instead of mse (default)"
    # parser.add_argument("--bce", help=help_, action='store_true')
    # parser.add_argument("--cp2", help=help_)
    # parser.add_argument("--cpz2", help=help_)
    # parser.add_argument("--cpz2LC", help=help_)
    # parser.add_argument("--train", action='store_true')
    # args = parser.parse_args()

    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # if args.weights:
    #     filepath = os.path.join(save_dir, args.weights)
    #     vae.load_weights(filepath)
    #     # # 查看前128张图片的训练效果
    #     a128batch=parsed_image_dataset.take(1)
    #     a128batch_list = list(a128batch.as_numpy_iterator())
    #     testresult=vae.predict(a128batch)
    #     testresult = np.squeeze(testresult)
    #     digit_size = 64
    #     row = 2
    #     col = 10
    #     figure = np.zeros((digit_size * row, digit_size * col))
    #     for i in range(col):
    #         # import pdb
    #         # pdb.set_trace()
    #         figure[0:digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i+10]
    #         figure[1*digit_size:2*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i+10]
    #     plt.figure(figsize=(15, 15))
    #     plt.imshow(figure, cmap='Greys_r')
    #     plt.savefig('try_C3_epoch200.jpg')
    #     plt.close()
    #         # for j in range(row):
    #         #     figure[j * digit_size: (j + 1) * digit_size,
    #         #        i * digit_size: (i + 1) * digit_size] = digit
    # else:
    #     # train the autoencoder
    #     if args.cp2:
    #         filepath = os.path.join(save_dir, args.cp)
    #         vae.load_weights(filepath)
    #         vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallback(models),MyepochsaveCallback(save_dir,vae)])
    #         filepath = 'vae_jiajin_16w.h5'
    #         vae.save_weights(filepath)
        
    #     elif args.cpz2:
    #         filepath = os.path.join(save_dir, args.cpz2)
    #         vae.load_weights(filepath)
    #         vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])
    #         filepath = 'vae_jiajin_16wLC.h5'
    #         vae.save_weights(filepath)
    #     elif args.cpz2LC:
    #         filepath = os.path.join(save_dir, args.cpz2LC)
    #         vae.load_weights(filepath)
    #         vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])
    #         filepath = 'vae_jiajin_16wLC40907.h5'
    #         vae.save_weights(filepath)
    #     elif args.train:
    #         vae.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallbackz2(models),MyepochsaveCallback(save_dir,vae)])
    #         filepath = 'vae_jiajin_16wLC40907.h5'
    #         vae.save_weights(filepath)
    vae_gan_instance.fit(parsed_image_dataset,epochs=epochs,callbacks=MyPlotCallback_test(vae_gan_instance,a128batch))
