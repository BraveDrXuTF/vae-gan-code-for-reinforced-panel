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
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Concatenate, Add, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import IPython.display as display
import matplotlib
from tensorflow.python import training
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.nn_impl import weighted_cross_entropy_with_logits
from tensorflow.python.ops.variables import trainable_variables 
from tensorflow.keras.callbacks import TensorBoard
matplotlib.use('Agg') # 这样可以不显示图窗
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import time
import datetime

# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
# test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# # 按这种默认的写法，就会导致ValueError: Expected scalar shape, saw shape: (4,).
# # 原因很简单，默认的里面结构就是只能接收一个loss
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)

# class My_tensorboardcallback(TensorBoard):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def on_epoch_end(self, epoch, logs=None):
#         super(My_tensorboardcallback, self).on_epoch_end(epoch,logs)
#         writer = self._get_writer(self._validation_run_name)
#         with writer.as_default():



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

    filename = os.path.join('images', "digits_over_latent16w_epoch{0}vaeGANz2918.png".format(epoch))
    # display a 30x30 2D manifold of digits
    n = 25
    digit_size = image_size
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
    image2 = -tf.reshape(image1, [128,128,3])/255.0 + 1 # 黑1
    image2 = image2[:,:,0] #把这一行去掉就可以实现三通道图片的输入
    return image2

    


image_size = 128
tfrecord_list=glob.glob('data_random_news67/*.tfrecords')
# tfrecord_list = ['data/images_batch0.tfrecords','data/images_batch1.tfrecords']
image_list_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)
parsed_image_dataset = image_list_dataset.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_image_function),
    cycle_length=4)



batch_size = 67
# batch_num = 931
# 原來dataset可以這樣做
parsed_image_dataset = parsed_image_dataset.map(parse_imagestr2numpy)

# parsed_image_dataset = parsed_image_dataset.shard(3,1)

# parsed_image_dataset = parsed_image_dataset.shuffle(buffer_size=67)

parsed_image_dataset = parsed_image_dataset.batch(batch_size)

parsed_image_dataset = parsed_image_dataset.apply(tf.data.experimental.ignore_errors())
input_shape = (image_size, image_size, 1)

kernel_size = 3
filters = 16
latent_dim = 16 
epochs = 100

# 自此开始可以修改
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



def build_discriminator_with_teacher(filters=16):
    inputs = Input(shape = input_shape, name='dis_input')
    x = inputs
    z_teacher_input = Input(shape= (latent_dim,), name='z_teacher')
    z_teacher = Dropout(rate=0.5)(z_teacher_input) # z_teacher具有0.5的概率死掉
    z_embedding = Dense(1024, activation='linear', name='z_embbding_dis')(z_teacher)
    #3层卷积
    for i in range(3):
        filters *= 2
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)
    x = Flatten()(x)
    #(16*16*128--1024,对16*16*128层施加dropout)
    x = Dropout(rate=0.2)(x)
    x = Dense(1024,activation='relu')(x)

    # 对z_embedding和x进行加和操作

    x = add([x,z_embedding])

    x = Dense(1,activation='linear')(x)

    return Model(inputs=[inputs, z_teacher_input], outputs=x, name='discriminator')


def build_refiner():
    z = Input(shape = (latent_dim,), name='z_input')
    reconstructed_picture = Input(shape = (image_size,image_size,1), name='reconstructed_picture')
    # 试一下不采用denoise的
    # reconstructed_picture_denoised = tf.where(reconstructed_picture>0.78,reconstructed_picture,0)
    reconstructed_picture_denoised = reconstructed_picture
    # z_embedding = Dense(64*64,activation='selu')(z)
    # z_embedding = Dense(128*128,activation='selu')(z_embedding)
    # 首先实验简单的线性变换
    # z_drop = Dropout(rate=0.3)(z)
    z_drop = z
    z_embedding = Dense(image_size*image_size, activation='linear')(z_drop)
    z_embedding = Reshape((image_size,image_size,1))(z_embedding)

    refined_map = Concatenate()([reconstructed_picture_denoised, z_embedding])

    refined_picture = Conv2D(filters=1,kernel_size=1, activation='sigmoid')(refined_map)

    return Model([z,reconstructed_picture],refined_picture, name='Refiner')


# 是否需要把strategy加上
class VAER_GAN(keras.Model):
    def __init__(
        self,
        generator,
        discriminator,
        encoder,
        refiner
    ):
        super(VAER_GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.refiner = refiner
    def compile(
        self,
        encoder_optimizer,
        generator_optimizer,
        discriminator_optimizer,
        refiner_optimizer,
        reconstructed_loss,
        kl_loss,
        discriminator_loss,
        gen_about_discriminator_loss,
        ref_about_discriminator_loss,
    ):
        super(VAER_GAN, self).compile()
        self.encoder_optimizer = encoder_optimizer
        self.gen_optimizer = generator_optimizer
        self.disc_optimizer = discriminator_optimizer
        self.refiner_optimizer = refiner_optimizer


        self.reconstructed_loss = reconstructed_loss
        self.kl_loss = kl_loss
        self.discriminator_loss = discriminator_loss
        self.gen_about_discriminator_loss = gen_about_discriminator_loss
        self.ref_about_discriminator_loss = ref_about_discriminator_loss

    def train_step(self, one_batch_data):
        # input_image, target = one_batch_data
        real_img = one_batch_data
        gen_z = tf.random.normal([batch_size,latent_dim])
        with tf.GradientTape(persistent=True) as tape:

            # # photo to monet back to photo
            z_mean, z_log_var, z = self.encoder(real_img, training=True)
            reconstructed_img = self.generator(z, training=True)
            generated_img = self.generator(gen_z, training=True)


            refined_reconstructed = self.refiner([z, reconstructed_img], training=True)
            refined_generated = self.refiner([gen_z, generated_img], training=True)


            real_output = self.discriminator([real_img, z], training=True)
            reconstructed_output = self.discriminator([reconstructed_img, z], training=True)
            generated_output = self.discriminator([generated_img, gen_z], training=True)
            refined_reconstructed_output = self.discriminator([refined_reconstructed, z], training=True)
            refined_generated_output = self.discriminator([refined_generated, gen_z], training=True)


            reconstruction_loss = self.reconstructed_loss(real_img, reconstructed_img)
            kl_loss = self.kl_loss(z_mean,z_log_var)
            discriminator_loss = self.discriminator_loss(real_output, reconstructed_output, generated_output, refined_reconstructed_output, refined_generated_output)
            gen_about_discriminator_loss = self.gen_about_discriminator_loss(reconstructed_output,generated_output, refined_reconstructed_output, refined_generated_output)
            refined_loss = self.reconstructed_loss(real_img, refined_reconstructed)

            ref_about_discriminator_loss = self.ref_about_discriminator_loss(refined_reconstructed_output, refined_generated_output)
            # ref_about_discriminator_loss = 0 这个有没有必要置零
            # 难道我一直忽略gen_about_discriminator_loss + ref_about_discriminator_loss这两项？
            
            # vae_loss = reconstruction_loss + kl_loss + refined_loss + gen_about_discriminator_loss + ref_about_discriminator_loss # 原 vae
            vae_loss = reconstruction_loss + kl_loss + gen_about_discriminator_loss # 新的vae loss完全隔绝了refiner对于编码的影响
        # Calculate the gradients for generator and discriminator
        encoder_gradients = tape.gradient(vae_loss,self.encoder.trainable_variables)

        generator_gradients1 = tape.gradient(reconstruction_loss, self.generator.trainable_variables)

        generator_gradients2 = tape.gradient(gen_about_discriminator_loss, self.generator.trainable_variables)
        
        # 隔绝refiner对generator的影响
        # generator_gradients3 = tape.gradient(refined_loss, self.generator.trainable_variables)

        refiner_gradients1 = tape.gradient(refined_loss, self.refiner.trainable_variables)

        refiner_gradients2 = tape.gradient(ref_about_discriminator_loss, self.refiner.trainable_variables)

        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)



        self.encoder_optimizer.apply_gradients(zip(encoder_gradients,self.encoder.trainable_variables))

        self.gen_optimizer.apply_gradients(zip(generator_gradients1,self.generator.trainable_variables))

        self.gen_optimizer.apply_gradients(zip(generator_gradients2,self.generator.trainable_variables))

        # self.gen_optimizer.apply_gradients(zip(generator_gradients3,self.generator.trainable_variables))

        self.disc_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))

        self.refiner_optimizer.apply_gradients(zip(refiner_gradients1,self.refiner.trainable_variables))
        
        self.refiner_optimizer.apply_gradients(zip(refiner_gradients2,self.refiner.trainable_variables))


        return {
            "vae_loss": vae_loss,
            "kl_loss": kl_loss,
            "reconstruction_loss": reconstruction_loss,
            "discriminator_loss": discriminator_loss,
            "refine_loss": refined_loss
        }
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed_img = self.generator(z)
        # refined_img = self.refiner([z,reconstructed_img])
        return reconstructed_img
# 修2：优化器的策略
# 修3： 大修 探讨refiner的必要性
encoder_optimizer = tf.keras.optimizers.RMSprop()
generator_optimizer = tf.keras.optimizers.RMSprop()
discriminator_optimizer = tf.keras.optimizers.RMSprop()
refiner_optimizer = tf.keras.optimizers.RMSprop()

# 修1：损失函数的进一步探索

def reconstructed_loss(real_img,reconstructed_img):
    reconstructed_img = K.flatten(reconstructed_img)
    real_img = K.flatten(real_img) # 显示shape为(None,)
    difference = reconstructed_img-real_img
    weighted_img_difference = tf.multiply(K.square(difference),1+2*real_img)
    # weighted_img_difference = K.square(difference)
    return image_size*image_size*tf.reduce_mean(weighted_img_difference,axis=-1)





# 我在维度方面仍然存在疑惑
def kl_loss(z_mean,z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss
# # discriminator_loss使用的不是图片，而是关于图片的那些输出
# def discriminator_loss(real_output,reconstructed_output,generated_output):
#     loss1 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real_output),real_output)
#     loss2 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(reconstructed_output),reconstructed_output)
#     loss3 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated_output),generated_output)
#     return loss1 + loss2 + loss3

# discriminator_loss使用的不是图片，而是关于图片的那些输出
def discriminator_loss(real_output,reconstructed_output,generated_output,refined_reconstructed_output,refined_generated_output):
    loss1 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real_output),real_output)
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(reconstructed_output),reconstructed_output)
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated_output),generated_output)
    # 可以做对比试验 refiner能不能增强discriminator的效果
    loss4 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(refined_reconstructed_output),refined_reconstructed_output)
    loss5 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(refined_generated_output),refined_generated_output)

    return loss1 + loss2 + loss3 + loss4 + loss5


def gen_about_discriminator_loss(reconstructed_output,generated_output,refined_reconstructed_output,refined_generated_output):
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(reconstructed_output),reconstructed_output)
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated_output),generated_output) 
    # 可以做对比实验 refiner能不能增强generator的效果 ，如果不能就隔绝refiner对generator的影响
    # loss4 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(refined_reconstructed_output),refined_reconstructed_output)
    # loss5 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(refined_generated_output),refined_generated_output)
    loss4 = 0

    loss5 = 0
    return loss2 + loss3 + loss4 + loss5

def ref_about_discriminator_loss(refined_reconstructed_output,refined_generated_output): 
    loss4 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(refined_reconstructed_output),refined_reconstructed_output)
    loss5 = keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(refined_generated_output),refined_generated_output)
    return loss4 + loss5

# 实例化对象
encoder_instance = build_encoder()
generator_instance = build_decoder()
discriminator_instance = build_discriminator_with_teacher()
refiner_instance = build_refiner()


VAER_GAN_instance = VAER_GAN(generator_instance,discriminator_instance,encoder_instance,refiner_instance)


# 编译VAER_GAN
VAER_GAN_instance.compile(encoder_optimizer,generator_optimizer,discriminator_optimizer,refiner_optimizer,reconstructed_loss,kl_loss,discriminator_loss,gen_about_discriminator_loss,ref_about_discriminator_loss)



# 自此以下停止修改


def test_plot(model,a128batch,epoch):
    testresult=model.predict(a128batch)
    testresult = np.squeeze(testresult)
    # testresult = np.where(testresult>0.78,testresult,0)
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
    plt.savefig('images/test_vae_no_r_epoch{0}_{1}.jpg'.format(epoch,time.strftime("_%a_%b_%d_%H_%M_%S_%Y", time.localtime())))
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
        self.vae.save_weights(os.path.join(self.save_dir,'modelepoch_{0}z2vaerGAN{1}.h5'.format(epoch,time.strftime("_%a_%b_%d_%H_%M_%S_%Y", time.localtime()))))
         

if __name__ == '__main__':
    print(1)
    parsed_image_dataset
    a128batch = parsed_image_dataset.take(1)
    a128batch_list = list(a128batch.as_numpy_iterator())
    save_dir = './vae_cnn_weights'
    VAER_GAN_instance.built = True
    # VAER_GAN_instance.load_weights(os.path.join(save_dir,'modelepoch_36z2vaerGAN_Fri_Nov_26_17_14_39_2021.h5'))
    
    # VAER_GAN_instance.load_weights(os.path.join(save_dir,'modelepoch_31z2vaerGAN_Fri_Nov_26_17_30_46_2021.h5')) # 使用没有shuffle操作的预训练权重
    VAER_GAN_instance.load_weights(os.path.join(save_dir,'modelepoch_0z2vaerGAN_Fri_Nov_26_18_49_57_2021.h5')) # shuffle操作30余轮后的训练权重
    
    VAER_GAN_instance.fit(parsed_image_dataset,epochs=epochs,callbacks=[MyPlotCallback_test(VAER_GAN_instance,a128batch),MyepochsaveCallback(save_dir,VAER_GAN_instance)])







