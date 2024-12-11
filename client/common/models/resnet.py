import tensorflow as tf


from tensorflow.keras import layers,models

def residual_block(x, filters, kernel_size=3, stride=1):
    # Convolução 1
    shortcut = x  # A conexão de atalho
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Convolução 2
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Somar a entrada com a saída (conexão residual)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x