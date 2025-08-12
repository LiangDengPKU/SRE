
"""
@author: dengl
"""

import tensorflow as tf

def cell_DenseNet169(N_CLASSES, HEIGHT, WIDTH):

    baseModel = tf.keras.applications.DenseNet169(
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3)
    )
    
    f5 = baseModel.get_layer("conv5_block32_concat").output
    x = tf.keras.layers.GlobalMaxPool2D()(f5)

    outputs = tf.keras.layers.Dense(N_CLASSES)(x)

    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)
    
    return model