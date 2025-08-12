#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:20:32 2021

@author: dengl
"""
#import keras
import tensorflow as tf
#import numpy as np


#N_CLASSES = 2

# resnet = tf.keras.applications.ResNet152(weights="imagenet")
# resnet.summary()

# baseModel = tf.keras.applications.ResNet152(weights="imagenet",include_top=False,input_shape = (512, 512, 3))
# baseModel.summary()

def cell_ResNet(N_CLASSES,HEIGHT,WIDTH):
    # define ResNet50 model
    baseModel = tf.keras.applications.ResNet152(weights="imagenet",include_top=False,input_shape = (HEIGHT, WIDTH, 3))
    #model.summary()
    # get AMP layer weights
    f5 = baseModel.get_layer("conv5_block3_out").output
    
    #x = tf.keras.layers.GlobalAveragePooling2D()(f5)
    
    x = tf.keras.layers.GlobalMaxPool2D()(f5)
    
    outputs = tf.keras.layers.Dense(N_CLASSES)(x)

    model = tf.keras.Model(inputs = baseModel.input, outputs = outputs )
    return model


# model = cell_ResNet()

# model.summary()








