from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from json import decoder, encoder
import re
from tokenize import generate_tokens
from matplotlib.image import imread
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
from tensorflow.python.keras.backend import conv2d, zeros
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.nn_impl import weighted_cross_entropy_with_logits
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.keras.callbacks import TensorBoard
# matplotlib.use('Agg') # 这样可以不显示图窗
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import time
import datetime
import heapq
import pandas as pd
from PIL import Image

image_size = 128
data_path='gene_9000'
pic_list=glob.glob(os.path.join(data_path,'*.jpg'))
# random.shuffle(pic_list) #彻底打乱顺序
# 确保顺序
pic_list = sorted(pic_list,key=lambda info: (int(info[10:-4]), info[-4:]) )

# 画图
def plot_results1d(models, experiment_dim=0, latent_dim=16):
    """1d

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    decoder = models
    experiment_code = np.ones((1, latent_dim))*2

    n = 5
    digit_size = image_size
    figure = np.zeros((digit_size * n, digit_size * 1))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_dim_value = np.linspace(-10, 10, n)
    z3 = 0
    for i, xi in enumerate(grid_dim_value):
        experiment_code = np.zeros((1, latent_dim))
        experiment_code[0, experiment_dim] = xi
        x_decoded = decoder.predict(experiment_code)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               0:digit_size] = digit

    plt.figure(figsize=(15, 15))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(time.strftime(
        "_%a_%b_%d_%H_%M_%S_%Y", time.localtime())+'.jpg')
    plt.close()  # 这句话保证图像不会重叠


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


# 计算数据集特征
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

# 画图


def plot_results(models, batch):
    """Plots labels and MNIST digits as function 
        of 3-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    z3_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for z3 in z3_list:
        filename = os.path.join(
            'images', "digits_over_latent16w_ind{0}_z3is{1}.png".format(batch//500+1, z3))
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
        plt.close()  # 这句话保证图像不会重叠
# 画图


def plot_resultsz2(models, epoch):
    """Plots labels and MNIST digits as function 
        of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    decoder = models.generator

    filename = os.path.join(
        'images', "digits_over_latent16w_epoch{0}vaeGANz2918.png".format(epoch))
    # display a 30x30 2D manifold of digits
    n = 25
    digit_size = image_size
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    zero_pad = [1, -1]*((latent_dim-2)//2)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi] + zero_pad])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(12, 12))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    # plt.savefig(filename)
    plt.close()  # 这句话保证图像不会重叠


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
    image2 = -tf.reshape(image1, [128, 128, 3])/255.0 + 1  # 黑1
    image2 = image2[:, :, 0]  # 把这一行去掉就可以实现三通道图片的输入
    return image2
# def testFunc():


image_size = 128
# tfrecord_list=glob.glob('data_random_new/*.tfrecords')
tfrecord_list = glob.glob('gene9000_no_randoms/*.tfrecords')
# tfrecord_list = ['data/images_batch0.tfrecords','data/images_batch1.tfrecords']
image_list_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_list)
parsed_image_dataset = image_list_dataset.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_image_function),
                                                     cycle_length=4)


batch_size = 10
batch_num = 1124
# 原來dataset可以這樣做
parsed_image_dataset = parsed_image_dataset.map(parse_imagestr2numpy)

parsed_image_dataset = parsed_image_dataset.batch(batch_size)

input_shape = (image_size, image_size, 1)

kernel_size = 3
filters = 16
latent_dim = 16
epochs = 100

# VAE model = encoder + decoder
# build encoder model


def build_encoder(filters=32):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # 3层卷积
    for i in range(3):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
        filters *= 2

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
    inputs = Input(shape=input_shape, name='dis_input')
    x = inputs
    z_teacher_input = Input(shape=(latent_dim,), name='z_teacher')
    z_teacher = Dropout(rate=0.75)(z_teacher_input)  # z_teacher具有0.75的概率死掉
    z_embedding = Dense(1024, activation='linear',
                        name='z_embbding_dis')(z_teacher)
    # 3层卷积
    for i in range(3):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
    x = Flatten()(x)
    # (16*16*128--1024,对16*16*128层施加dropout)
    x = Dropout(rate=0.2)(x)
    x = Dense(1024, activation='relu')(x)

    # 对z_embedding和x进行加和操作

    x = add([x, z_embedding])

    x = Dense(1, activation='linear')(x)

    return Model(inputs=[inputs, z_teacher_input], outputs=x, name='discriminator')


def build_refiner():
    z = Input(shape=(latent_dim,), name='z_input')
    reconstructed_picture = Input(
        shape=(image_size, image_size, 1), name='reconstructed_picture')
    reconstructed_picture_denoised = tf.where(
        reconstructed_picture > 0.78, reconstructed_picture, 0)
    # z_embedding = Dense(64*64,activation='selu')(z)
    # z_embedding = Dense(128*128,activation='selu')(z_embedding)
    # 首先实验简单的线性变换
    z_embedding = Dense(image_size*image_size, activation='linear')(z)
    z_embedding = Reshape((image_size, image_size, 1))(z_embedding)

    refined_map = Concatenate()([reconstructed_picture_denoised, z_embedding])

    refined_picture = Conv2D(filters=1, kernel_size=1,
                             activation='sigmoid')(refined_map)

    return Model([z, reconstructed_picture], refined_picture, name='Refiner')

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

    def train_step(self, one_batch_data):
        # input_image, target = one_batch_data
        real_img = one_batch_data
        gen_z = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape(persistent=True) as tape:

            # # photo to monet back to photo
            z_mean, z_log_var, z = self.encoder(real_img, training=True)
            reconstructed_img = self.generator(z, training=True)
            generated_img = self.generator(gen_z, training=True)

            refined_reconstructed = self.refiner(
                [z, reconstructed_img], training=True)
            refined_generated = self.refiner(
                [gen_z, generated_img], training=True)

            real_output = self.discriminator([real_img, z], training=True)
            reconstructed_output = self.discriminator(
                [reconstructed_img, z], training=True)
            generated_output = self.discriminator(
                [generated_img, gen_z], training=True)
            refined_reconstructed_output = self.discriminator(
                [refined_reconstructed, z], training=True)
            refined_generated_output = self.discriminator(
                [refined_generated, gen_z], training=True)

            reconstruction_loss = self.reconstructed_loss(
                real_img, reconstructed_img)
            kl_loss = self.kl_loss(z_mean, z_log_var)
            discriminator_loss = self.discriminator_loss(
                real_output, reconstructed_output, generated_output, refined_reconstructed_output, refined_generated_output)
            gen_about_discriminator_loss = self.gen_about_discriminator_loss(
                reconstructed_output, generated_output, refined_reconstructed_output, refined_generated_output)
            refined_loss = self.reconstructed_loss(
                real_img, refined_reconstructed)

            vae_loss = reconstruction_loss + kl_loss + refined_loss

        # Calculate the gradients for generator and discriminator
        encoder_gradients = tape.gradient(
            vae_loss, self.encoder.trainable_variables)

        generator_gradients1 = tape.gradient(
            reconstruction_loss, self.generator.trainable_variables)

        generator_gradients2 = tape.gradient(
            gen_about_discriminator_loss, self.generator.trainable_variables)

        generator_gradients3 = tape.gradient(
            refined_loss, self.generator.trainable_variables)

        refiner_gradients1 = tape.gradient(
            refined_loss, self.refiner.trainable_variables)

        refiner_gradients2 = tape.gradient(
            refined_loss, self.refiner.trainable_variables)

        discriminator_gradients = tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables)

        self.encoder_optimizer.apply_gradients(
            zip(encoder_gradients, self.encoder.trainable_variables))

        self.gen_optimizer.apply_gradients(
            zip(generator_gradients1, self.generator.trainable_variables))

        self.gen_optimizer.apply_gradients(
            zip(generator_gradients2, self.generator.trainable_variables))

        self.gen_optimizer.apply_gradients(
            zip(generator_gradients3, self.generator.trainable_variables))

        self.disc_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        self.refiner_optimizer.apply_gradients(
            zip(refiner_gradients1, self.refiner.trainable_variables))

        self.refiner_optimizer.apply_gradients(
            zip(refiner_gradients2, self.refiner.trainable_variables))

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


encoder_optimizer = tf.keras.optimizers.RMSprop()
generator_optimizer = tf.keras.optimizers.RMSprop()
discriminator_optimizer = tf.keras.optimizers.RMSprop()
refiner_optimizer = tf.keras.optimizers.RMSprop()


def reconstructed_loss(real_img, reconstructed_img):
    reconstructed_img = K.flatten(reconstructed_img)
    real_img = K.flatten(real_img)  # 显示shape为(None,)
    difference = reconstructed_img-real_img
    # weighted_img_difference = tf.multiply(K.square(difference),0.625+0.625*real_img)
    weighted_img_difference = K.square(difference)
    return image_size*image_size*tf.reduce_mean(weighted_img_difference, axis=-1)


# 我在维度方面仍然存在疑惑
def kl_loss(z_mean, z_log_var):
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


def discriminator_loss(real_output, reconstructed_output, generated_output, refined_reconstructed_output, refined_generated_output):
    loss1 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(real_output), real_output)
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(reconstructed_output), reconstructed_output)
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(generated_output), generated_output)
    loss4 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(refined_reconstructed_output), refined_reconstructed_output)
    loss5 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(refined_generated_output), refined_generated_output)

    return loss1 + loss2 + loss3 + loss4 + loss5


def gen_about_discriminator_loss(reconstructed_output, generated_output, refined_reconstructed_output, refined_generated_output):
    loss2 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(reconstructed_output), reconstructed_output)
    loss3 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(generated_output), generated_output)
    loss4 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(refined_reconstructed_output), refined_reconstructed_output)
    loss5 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(refined_generated_output), refined_generated_output)
    return loss2 + loss3 + loss4 + loss5


def ref_about_discriminator_loss(refined_reconstructed_output, refined_generated_output):
    loss4 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(refined_reconstructed_output), refined_reconstructed_output)
    loss5 = keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(refined_generated_output), refined_generated_output)
    return loss4 + loss5


# 实例化对象
encoder_instance = build_encoder()
generator_instance = build_decoder()
discriminator_instance = build_discriminator_with_teacher()
refiner_instance = build_refiner()


VAER_GAN_instance = VAER_GAN(
    generator_instance, discriminator_instance, encoder_instance, refiner_instance)


# 编译VAER_GAN
VAER_GAN_instance.compile(encoder_optimizer, generator_optimizer, discriminator_optimizer,
                          refiner_optimizer, reconstructed_loss, kl_loss, discriminator_loss, gen_about_discriminator_loss)


VAER_GAN_instance.built = True
# VAER_GAN_instance.load_weights(
#     'vae_cnn_weights/modelepoch_4z2vaerGAN_Sat_Sep_25_11_34_31_2021.h5')

VAER_GAN_instance.load_weights('vae_cnn_weights/modelepoch_50z2vaerGAN_Fri_Nov_26_19_11_10_2021.h5')
a128batch = parsed_image_dataset.take(1)
print(type(a128batch))
a128batch_list = list(a128batch.as_numpy_iterator())


def fine_tuning_z(z):

    plt.imshow(np.squeeze(VAER_GAN_instance.generator.predict(z)), cmap='Greys_r')
    plt.show()
    for i in range(1, 100):
        z[0] = z[0] + 1e-1
        plt.imshow(np.squeeze(
            VAER_GAN_instance.generator.predict(z)), cmap='Greys_r')
        plt.show()


def compare_z(z1, z2, transitional_num=10):
    # distance = np.linalg.norm(z1,z2)
    plt.imshow(np.squeeze(VAER_GAN_instance.generator.predict(z1)),
               cmap='Greys_r')
    for i in range(transitional_num):

        z = z1 + (z2 - z1)*(i + 1)/transitional_num

        plt.imshow(np.squeeze(
            VAER_GAN_instance.generator.predict(z)), cmap='Greys_r')

        plt.show()

# 问题：check_index=70时有很多很像
def collect_z(check_index=410, check_num=6):

    _, _, code = VAER_GAN_instance.encoder.predict(parsed_image_dataset)

    check_img = Image.open('gene_9000/{}.jpg'.format(check_index))
    check_img = np.array(check_img)

    print(-(check_img/255.0))
    check_img = -(check_img/255.0) + 1
    print(check_img)
    plt.imshow(check_img, cmap='Greys_r')
    plt.savefig('orig__{}.jpg'.format(time.strftime("_%a_%b_%d_%H_%M_%S_%Y", time.localtime())))
    plt.show()
    check_img = check_img[None, :, :, 0]

   # print(check_img.shape)
    _, _, check_code = VAER_GAN_instance.encoder.predict(check_img)
    print(check_code)
    codelist = []
    # numpy array转换成列表
    for i in range(len(code)):
        codelist.append(code[None, i])

    # check_code = codelist[check_index]

    plt.imshow(np.squeeze(VAER_GAN_instance.generator.predict(
        check_code)), cmap='Greys_r')
    plt.show()

    codelist.pop(check_index)
    distance_list = []
    for i, other_code in enumerate(codelist):
        # print(type(check_code),type(other_code))
        diff = check_code-other_code
        distance_list.append(np.linalg.norm(diff))
    result_index = map(distance_list.index,
                       heapq.nsmallest(check_num, distance_list))
    result_index = list(result_index)
    print(result_index)
    # result用于存储最近的code
    # result_list用于存储最近的code下标
    result = []
    for i in range(check_num):

        result.append(codelist[result_index[i]])
        print('{}--{}'.format(result_index[i], distance_list[i]))

    for i in range(check_num):

        plt.imshow(np.squeeze(
            VAER_GAN_instance.generator.predict(result[i])), cmap='Greys_r')
        plt.savefig('{}___{}.jpg'.format(time.strftime("_%a_%b_%d_%H_%M_%S_%Y", time.localtime()),i))
        plt.show()
        


def test_plot(model, a128batch):
    testresult = model.predict(a128batch)
    testresult = np.squeeze(testresult)
    # testresult = np.where(testresult>0.78,testresult,0)
    digit_size = 128
    row = 2
    col = 2
    figure = np.zeros((digit_size * 2 * row, digit_size * col))  # 4行10列
    for i in range(col):
        # import pdb
        # pdb.set_trace()
        figure[0:digit_size, i *
               digit_size: (i + 1) * digit_size] = a128batch_list[0][i]
        figure[1*digit_size:2*digit_size, i *
               digit_size: (i + 1) * digit_size] = testresult[i]
        figure[2*digit_size:3*digit_size, i *
               digit_size: (i + 1) * digit_size] = a128batch_list[0][i + int(batch_size/2)]
        figure[3*digit_size:4*digit_size, i *
               digit_size: (i + 1) * digit_size] = testresult[i + int(batch_size/2)]
    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    # plt.savefig('images/test_vaer_epoch{0}_{1}.jpg'.format(epoch,time.strftime("_%a_%b_%d_%H_%M_%S_%Y", time.localtime())))
    plt.close()

def to_excel():

    code_list = []
    for i in range(len(pic_list)):
        img = Image.open(pic_list[i])
        img = np.array(img)
        img = -(img/255.0) + 1
        img = img[None, :, :, 0]
        _, _, img_code = VAER_GAN_instance.encoder.predict(img) # img_code的shape是1，16，np.array类型
        img_code = np.squeeze(img_code) 
        img_code = img_code.tolist()
        # img_code.tolist()
        # 做这些操作的目的都是让code_list能顺利转换成code:shape (9313,16)
        code_list.append(img_code)
    
    # _, _, code = VAER_GAN_instance.encoder.predict(parsed_image_dataset)
    code = np.array(code_list)
    code_df = pd.DataFrame(code)
    data_index = pic_list
    code_df.index = data_index
    code_df.to_excel('code.xlsx','sheet_1',float_format='%f')
def to_excel_special():
    special_pic_list = glob.glob('Special/*.jpg')
    special_pic_list = sorted(special_pic_list, key=lambda info: (int(info[8:-4]), info[-4:]) )
    list_to_excel(special_pic_list)
def list_to_excel(pic_list):
    code_list = []
    for i in range(len(pic_list)):
        img = Image.open(pic_list[i])
        img = np.array(img)
        img = -(img/255.0) + 1
        img = img[None, :, :, 0]
        _, _, img_code = VAER_GAN_instance.encoder.predict(img) # img_code的shape是1，16，np.array类型
        img_code = np.squeeze(img_code) 
        img_code = img_code.tolist()
        # img_code.tolist()
        # 做这些操作的目的都是让code_list能顺利转换成code:shape (9313,16)
        code_list.append(img_code)
    
    # _, _, code = VAER_GAN_instance.encoder.predict(parsed_image_dataset)
    code = np.array(code_list)
    code_df = pd.DataFrame(code)
    data_index = pic_list
    code_df.index = data_index
    code_df.to_excel('code_special.xlsx','sheet_1',float_format='%f')

if __name__ == '__main__':

    # z_mean, z_log_var, z_a128 = VAER_GAN_instance.encoder.predict(a128batch)

    # z_a128保持维度的切片
    # print(z_a128[None,6].shape)
    # # print(z_a128)
    # plt.imshow(z_a128[1], cmap='Greys_r')
    # plt.show()
    # fine_tuning_z(z_a128[None,0])
    # compare_z(z_a128[None,5],z_a128[None,7])
    # print(z_a128[None,0],z_a128[None,1])
    # collect_z(check_index=224)
    # plot_resultsz2(VAER_GAN_instance,0)

    # test_plot(VAER_GAN_instance, a128batch)  # 这一关过了
    # collect_z(check_index=0,check_num=5)
    # test_plot(VAER_GAN_instance,a128batch)
    # to_excel()
    # a=[3.133805,-2.444599,0.725415,2.12828,-3.216603,-1.565976,2.288015,1.007065,3.84931,0.489287,-1.598607,-0.799177,1.553569,3.607766,-1.416075,2.568722]
    # a=[0.944523,0.517405,1.705929,1.130903,-5.166395,-0.089059,3.144279,-1.535654,2.833469,-1.696541,-2.188113,0.465848,0.612227,2.279999,1.529394,2.096762
    # ]
    
    # a = np.array(a)
    # a = np.expand_dims(a,axis=0)
    # # VAER_GAN_instance.generator.predict(a)
    # plt.imshow(np.squeeze(
    #         VAER_GAN_instance.generator.predict(a)), cmap='Greys_r')
    # plt.savefig(time.strftime("hahaha_%a_%b_%d_%H_%M_%S_%Y", time.localtime())+'.jpg')
    to_excel_special()
