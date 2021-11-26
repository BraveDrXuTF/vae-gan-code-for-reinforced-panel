# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:33:19 2021

@author: Xutf
"""
import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
def plot_results(models,
                 batch_size=128,
                 model_name="vae_jiajin"):
    """Plots labels and MNIST digits as function 
        of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    # x_test, y_test = data
    # xmin = ymin = -4
    # xmax = ymax = +4
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean1.png")
    # display a 2D plot of the digit classes in the latent space
    # z, _, _ = encoder.predict(x_test,
    #                           batch_size=batch_size)
    # plt.figure(figsize=(12, 10))

    # axes x and y ranges
    # axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    # axes.set_ylim([ymin,ymax])

    # subsample to reduce density of points on the plot
    # z = z[0::2]
    # y_test = y_test[0::2]
    # plt.scatter(z[:, 0], z[:, 1], marker="")
    # for i, digit in enumerate(y_test):
    #     axes.annotate(digit, (z[i, 0], z[i, 1]))
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    filename = os.path.join(model_name, "digits_over_latent16w_z3_0.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 64
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    zi = 0
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi, zi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    # 这里不注释掉会报错
    # start_range = digit_size // 2
    # end_range = n * digit_size + start_range + 1
    # pixel_range = np.arange(start_range, end_range, digit_size)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()
        
