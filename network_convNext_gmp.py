
"""
@author: mediway
"""

import tensorflow as tf


def cell_ConvNeXt_256(N_CLASSES, HEIGHT=256, WIDTH=256):
    baseModel = tf.keras.applications.convnext.ConvNeXtTiny(
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3)
    )
    
    x = baseModel.output
    x = tf.keras.layers.GlobalMaxPool2D()(x)
    outputs = tf.keras.layers.Dense(N_CLASSES)(x)    
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)    
    return model


def cell_ConvNeXt_flexible(N_CLASSES, HEIGHT, WIDTH):
    """
    - Tiny/Small: 224, 256, 384, 512
    - Base/Large: 224, 256, 384, 512
    """

    min_dim = 32  
    if HEIGHT < 32 or WIDTH < 32:
        raise ValueError(f">=32×32，inputing {HEIGHT}×{WIDTH}")
    
    if HEIGHT % 32 != 0 or WIDTH % 32 != 0:
        print(f"warning：Input size {HEIGHT}×{WIDTH} should be divisible by 32.")
    
    baseModel = tf.keras.applications.convnext.ConvNeXtBase(
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3)
    )
    
    x = baseModel.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)   
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)    
    return model


def cell_ConvNeXt_256_optimized(N_CLASSES):

    baseModel = tf.keras.applications.convnext.ConvNeXtBase(
        weights="imagenet",
        include_top=False,
        input_shape=(256, 256, 3)
    )
    
    x = baseModel.output
    x = tf.keras.layers.Conv2D(512, (1, 1), padding='same')(x)  # 降维
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # classification head
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)   
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)   
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)   
    return model




# test
if __name__ == "__main__":
    print("256×256:")
    model_256 = cell_ConvNeXt_256(6, 256, 256)
    model_256.summary()
    
    print("\n224×224 testing...")
    model_224 = cell_ConvNeXt_256(6, 224, 224)
    
    print("\n flexible test...")
    model_flex = cell_ConvNeXt_flexible(6, 256, 256)
    
    # test forward
    import numpy as np
    test_input = np.random.randn(1, 256, 256, 3).astype(np.float32)
    output = model_256.predict(test_input, verbose=0)
    print(f"\ninput shape: {test_input.shape}, output shape: {output.shape}")