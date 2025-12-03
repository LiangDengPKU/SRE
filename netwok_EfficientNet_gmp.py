
"""
@author: Liang Deng
"""

import tensorflow as tf

def cell_EfficientNetB2(N_CLASSES, HEIGHT, WIDTH):
    baseModel = tf.keras.applications.EfficientNetB2(
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3)
    )
    last_feature_layer = baseModel.get_layer("top_activation").output    
    x = tf.keras.layers.GlobalMaxPool2D()(last_feature_layer)   
    outputs = tf.keras.layers.Dense(N_CLASSES)(x)    
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)    
    return model


def cell_EfficientNetB2_enhanced(N_CLASSES, HEIGHT, WIDTH):

    baseModel = tf.keras.applications.EfficientNetB2(
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3)
    )
    
    shallow_features = baseModel.get_layer("block3a_expand_activation").output    
    mid_features = baseModel.get_layer("block5a_expand_activation").output   
    deep_features = baseModel.get_layer("top_activation").output
    
    # deep layer
    x_deep = tf.keras.layers.GlobalMaxPool2D()(deep_features)
    x_deep = tf.keras.layers.Dense(512, activation='relu')(x_deep)
    
    # mid layer
    x_mid = tf.keras.layers.GlobalMaxPool2D()(mid_features)
    x_mid = tf.keras.layers.Dense(256, activation='relu')(x_mid)
    
    # shallow layer
    x_shallow = tf.keras.layers.GlobalMaxPool2D()(shallow_features)
    x_shallow = tf.keras.layers.Dense(128, activation='relu')(x_shallow)
    
    # fusion
    x = tf.keras.layers.Concatenate()([x_deep, x_mid, x_shallow])    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(N_CLASSES)(x)   
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)    
    return model



def cell_EfficientNet(N_CLASSES, HEIGHT, WIDTH, version='B2'):
    """
        version: 'B0'到'B7'，默认'B2'
    """
    efficientnet_models = {
        'B0': tf.keras.applications.EfficientNetB0,
        'B1': tf.keras.applications.EfficientNetB1,
        'B2': tf.keras.applications.EfficientNetB2,
        'B3': tf.keras.applications.EfficientNetB3,
        'B4': tf.keras.applications.EfficientNetB4,
        'B5': tf.keras.applications.EfficientNetB5,
        'B6': tf.keras.applications.EfficientNetB6,
        'B7': tf.keras.applications.EfficientNetB7,
    }
    
    if version not in efficientnet_models:
        raise ValueError(f"support: {list(efficientnet_models.keys())}")
    
    # 加载指定版本的EfficientNet
    baseModel = efficientnet_models[version](
        weights="imagenet",
        include_top=False,
        input_shape=(HEIGHT, WIDTH, 3)
    )
    try:
        feature_layer = baseModel.get_layer("top_activation")
    except:
        feature_layer = baseModel.layers[-1]
    
    x = tf.keras.layers.GlobalMaxPool2D()(feature_layer.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)    
    outputs = tf.keras.layers.Dense(N_CLASSES)(x)   
    model = tf.keras.Model(inputs=baseModel.input, outputs=outputs)   
    return model

# test
if __name__ == "__main__":
    print("EfficientNet B2:")
    model_simple = cell_EfficientNetB2(N_CLASSES=6, HEIGHT=224, WIDTH=224)
    model_simple.summary()
    
    print("\nEfficientNet B2 enhanced tested")
    model_enhanced = cell_EfficientNetB2_enhanced(N_CLASSES=6, HEIGHT=224, WIDTH=224)
    
    print("\nEfficientNet B3 tested")
    model_general = cell_EfficientNet(N_CLASSES=6, HEIGHT=224, WIDTH=224, version='B3')
    
    # forward test
    import numpy as np
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    output = model_simple.predict(test_input, verbose=0)
    print(f"\nforward test - output shape: {output.shape}")