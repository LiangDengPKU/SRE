
"""
@author: dengl
"""

import tensorflow as tf

def cell_ResNet(N_CLASSES,HEIGHT,WIDTH):
    baseModel = tf.keras.applications.ResNet152(weights="imagenet",include_top=False,input_shape = (HEIGHT, WIDTH, 3))
    #model.summary()
    # get AMP layer weights
    f5 = baseModel.get_layer("conv5_block3_out").output  
    x = tf.keras.layers.GlobalMaxPool2D()(f5)
    outputs = tf.keras.layers.Dense(N_CLASSES)(x)
    model = tf.keras.Model(inputs = baseModel.input, outputs = outputs )
    return model
