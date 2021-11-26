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


kernel_size = 3
latent_dim = 16
batch_size = 128
epochs = 100

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
    image2 = -tf.reshape(image1, [64,64,3])/255.0 + 1 #黑1
    image2 = image2[:,:,0] #把这一行去掉就可以实现三通道图片的输入
    return image2
    


image_size = 64
tfrecord_list=glob.glob('data_random/*.tfrecords')
# tfrecord_list = ['data/images_batch0.tfrecords','data/images_batch1.tfrecords']
image_list_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)
parsed_image_dataset = image_list_dataset.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_image_function),
    cycle_length=4)



batch_size = 128
batch_num = 1302
# 原來dataset可以這樣做
parsed_image_dataset = parsed_image_dataset.map(parse_imagestr2numpy)

parsed_image_dataset = parsed_image_dataset.batch(128)





def build_generator(filters=128):
    # build decoder model
    latent_inputs = Input(shape=(64,), name='z_sampling')

    x = latent_inputs

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

# discriminator_loss使用的不是图片，而是关于图片的那些输出
def discriminator_loss(real_output,generated_output):
    loss1 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real_output),real_output)
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated_output),generated_output)
    return loss1  + loss2

def gen_about_discriminator_loss(generated_output):
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated_output),generated_output) 
    return loss3  


# 是否需要把strategy加上
class GAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
    ):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        discriminator_loss,
        gen_about_discriminator_loss
    ):
        super(GAN, self).compile()

        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer
        self.discriminator_loss = discriminator_loss
        self.gen_about_discriminator_loss = gen_about_discriminator_loss
    def train_step(self, one_batch_data):
        # input_image, target = one_batch_data
        real_img = one_batch_data
        gen_z = tf.random.normal((latent_dim,))
        with tf.GradientTape(persistent=True) as tape:

            # # photo to monet back to photo

            generated_img = self.generator(gen_z, training=True)
            real_output = self.discriminator(real_img, training=True)
            generated_output = self.discriminator(generated_img, training=True)
            discriminator_loss = self.discriminator_loss(real_output,generated_output)

            gen_about_discriminator_loss = self.gen_about_discriminator_loss(generated_output)
        # Calculate the gradients for generator and discriminator


        generator_gradients2 = tape.gradient(gen_about_discriminator_loss, self.generator.trainable_variables)

        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)




        self.gen_optimizer.apply_gradients(zip(generator_gradients2,self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

        
        return {

            "gen_about_discriminator_loss": gen_about_discriminator_loss,
            "discriminator_loss": discriminator_loss
        }
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed_img = self.generator(z)
        return reconstructed_img


gen_instance = build_generator()
dis_instance = build_discriminator()
gan_instance = GAN(gen_instance,dis_instance)


generator_optimizer = tf.keras.optimizers.RMSprop()
discriminator_optimizer = tf.keras.optimizers.RMSprop()

# 编译vae_gan
gan_instance.compile(generator_optimizer,discriminator_optimizer,discriminator_loss, gen_about_discriminator_loss)



if __name__ == '__main__':
    gan_instance.fit(parsed_image_dataset,epochs=epochs)