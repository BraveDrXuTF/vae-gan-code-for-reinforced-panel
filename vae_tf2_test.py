# -*- coding: utf-8 -*-
"""
Created on Sun May  9 09:59:57 2021

@author: Administrator
"""
from vae_tf2 import plot_results
from vae_cnn_mnist import build_vae

model_path = 'vae_jiajin_2000.h5'
vae,encoder,decoder = build_vae()
vae.load_weights(model_path)

