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
from tensorflow.keras import callbacks

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
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
image_size = 128
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
        plt.close() # 这句话保证图像不会重叠
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

    filename = os.path.join('images', "digits_over_latent16w_epoch{0}GAN614.png".format(epoch))
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
    image2 = -tf.reshape(image1, [128,128,3])/255.0 + 1 # 原来是白底的，现在我改成黑底的，筋条部分为1（白色），背景为0（黑色）
    image2 = image2[:,:,0] #把这一行去掉就可以实现三通道图片的输入
    return image2

    



tfrecord_list=glob.glob('data_random_new/*.tfrecords')
# tfrecord_list = ['data/images_batch0.tfrecords','data/images_batch1.tfrecords']
image_list_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)
parsed_image_dataset = image_list_dataset.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_image_function),
    cycle_length=4)



batch_size = 4
batch_num = 1124
# 原來dataset可以這樣做
parsed_image_dataset = parsed_image_dataset.map(parse_imagestr2numpy)

parsed_image_dataset = parsed_image_dataset.batch(batch_size=batch_size)

input_shape = (image_size, image_size, 1)

kernel_size = 3
filters = 16
latent_dim = 2 
epochs = 100

# VAE model = encoder + decoder
# build encoder model
def build_encoder(filters=32):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    #3层卷积
    for i in range(3):
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
        filters*=2

    # shape info needed to build decoder model 
    # 2 latent vector
    shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(latent_dim, activation=None, name='x')(x)
    z_mean = Dense(latent_dim, name='z_mean', activation=None)(x)
    z_log_var = Dense(latent_dim, name='z_log_var', activation=None)(x)

    z = Lambda(sampling,
            output_shape=(latent_dim,), 
            name='z')([z_mean, z_log_var])

    # instantiate encoder model
    return Model(inputs, [z_mean, z_log_var, z], name='encoder')
    

def build_decoder(filters=128):
    # build decoder model
    shape = (None, 16, 16, 128)
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
    # 经过debug ,确认为64*64*1
    outputs = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)

    # instantiate decoder model
    return Model(latent_inputs, outputs, name='decoder_or_generator')





# 是否需要把strategy加上
class VAE(keras.Model):
    def __init__(
        self,
        generator,
        encoder
    ):
        super(VAE, self).__init__()
        self.generator = generator

        self.encoder = encoder
    def compile(
        self,
        encoder_optimizer,
        generator_optimizer,
        reconstructed_loss,
        kl_loss
    ):
        super(VAE, self).compile()
        self.encoder_optimizer = encoder_optimizer
        self.gen_optimizer = generator_optimizer
        self.reconstructed_loss = reconstructed_loss
        self.kl_loss = kl_loss

    def train_step(self, one_batch_data):
        # input_image, target = one_batch_data
        real_img = one_batch_data

        with tf.GradientTape(persistent=True) as tape:

            # # photo to monet back to photo
            z_mean, z_log_var, z = self.encoder(real_img, training=True)
            reconstructed_img = self.generator(z, training=True)



            
            
            reconstruction_loss = self.reconstructed_loss(real_img,reconstructed_img)
            kl_loss = self.kl_loss(z_mean,z_log_var)


            vae_loss = reconstruction_loss + kl_loss

        # Calculate the gradients for generator and discriminator
        encoder_gradients = tape.gradient(vae_loss,self.encoder.trainable_variables)

        generator_gradients1 = tape.gradient(reconstruction_loss, self.generator.trainable_variables)




        self.encoder_optimizer.apply_gradients(zip(encoder_gradients,self.encoder.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(generator_gradients1,self.generator.trainable_variables))


        
        return {
            "vae_loss": vae_loss,
            "kl_loss": kl_loss,
            "reconstruction_loss": reconstruction_loss,
        }

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed_img = self.generator(z)
        return reconstructed_img


encoder_optimizer = tf.keras.optimizers.RMSprop()
generator_optimizer = tf.keras.optimizers.RMSprop()


def reconstructed_loss(real_img,reconstructed_img):
    reconstructed_img = K.flatten(reconstructed_img)
    real_img = K.flatten(real_img) # 显示shape为(None,)
    difference = reconstructed_img-real_img
    weighted_img_difference = tf.multiply(K.square(difference),0.75+1.5*real_img)
    return image_size*image_size*tf.reduce_mean(weighted_img_difference,axis=-1)





# 我在维度方面仍然存在疑惑
def kl_loss(z_mean,z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

 




# 实例化对象
encoder_instance = build_encoder()
generator_instance = build_decoder()

vae_instance = VAE(generator=generator_instance,encoder=encoder_instance)


def test_plot(model,a128batch,epoch):
    testresult=model.predict(a128batch)
    testresult = np.squeeze(testresult)
    digit_size = 128
    row = 2
    col = 2
    figure = np.zeros((digit_size * 2 *row, digit_size * col)) # 4行10列
    for i in range(col):
        # import pdb
        # pdb.set_trace()
        figure[0:digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i]
        figure[1*digit_size:2*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i]
        figure[2*digit_size:3*digit_size,i * digit_size: (i + 1) * digit_size] = a128batch_list[0][i + int(batch_size/2)]
        figure[3*digit_size:4*digit_size,i * digit_size: (i + 1) * digit_size] = testresult[i + int(batch_size/2)]
    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('images/test_vae918_1_epoch{0}.jpg'.format(epoch))
    plt.close()

class MyPlotCallback_test(Callback):
    def __init__(self, model, a128batch):
        self.model = model
        self.a128batch = a128batch
    def on_epoch_end(self, epoch, logs=None):
        test_plot(self.model, self.a128batch, epoch)
class MyepochsaveCallback(Callback):
    def __init__(self, save_dir, vae):
        self.save_dir = save_dir
        self.vae=vae
    def on_epoch_end(self, epoch, logs=None):
        self.vae.save_weights(os.path.join(self.save_dir,'modelepoch_{0}z2VAE918_1.h5'.format(epoch)))



# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = {}
 
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))


if __name__ == '__main__':
    a128batch = parsed_image_dataset.take(1)
    a128batch_list = list(a128batch.as_numpy_iterator())
    save_dir = './vae_cnn_weights'
    vae_instance.compile(encoder_optimizer=encoder_optimizer,generator_optimizer=generator_optimizer,reconstructed_loss=reconstructed_loss, kl_loss=kl_loss)
    vae_instance.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyepochsaveCallback(save_dir,vae_instance),MyPlotCallback_test(vae_instance,a128batch)])
