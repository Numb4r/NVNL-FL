import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def CNN_fmnist_tf(input_shape, num_classes):
    model = models.Sequential()
    
    # Layer 1
    model.add(layers.Conv2D(16, kernel_size=(5, 5), padding='same', input_shape=(input_shape[1:])))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 2
    model.add(layers.Conv2D(32, kernel_size=(3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Layer 3
    model.add(layers.Conv2D(64, kernel_size=(3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def residual_block(inputs, filters, stride=1):
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same", use_bias=False)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def ResNet20(input_shape, num_classes):
    inputs = tf.keras.Input(shape=(input_shape[1:]))

    # Initial Conv Layer
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # First stack of residual blocks (16 filters)
    for _ in range(3):
        x = residual_block(x, 16)

    # Second stack of residual blocks (32 filters)
    x = residual_block(x, 32, stride=2)
    for _ in range(2):
        x = residual_block(x, 32)

    # Third stack of residual blocks (64 filters)
    x = residual_block(x, 64, stride=2)
    for _ in range(2):
        x = residual_block(x, 64)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_dnn(input_shape, num_classes):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape[1:])),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64,  activation='relu'),
        tf.keras.layers.Dense(32,  activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_lenet5(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(*input_shape[1:],1), padding="same"))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='tanh'))
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
   # Convolutional Block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Convolutional Block 2
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Convolutional Block 3
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    
    # Fully Connected Layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def flat_parameters(parameters):
    flat_params = []

    for param in parameters:
        flat_params.extend(param.flatten())

    return flat_params


def reshape_parameters(self, decrypted_parameters):
    reshaped_parameters = []

    for layer in self.model.get_weights():
        reshaped_parameters.append(np.reshape(decrypted_parameters[:layer.size], layer.shape))
        decrypted_parameters = decrypted_parameters[layer.size:]

    return reshaped_parameters
def reshape_model(flatted_packs,a,sentinel=0):
    model =[]
    # print(a)
    
    if type(a) != int and type(a)!=float:
        # print(f'len:{len(a)}')
        for item in a:
            # print(f'item {item},  {type(item)},sentinel {sentinel}')
            
            m,sentinel=reshape_model(flatted_packs,item,sentinel)
            # print(sentinel)
            model.append(m)
    else:
    
        sentinel+=1 
        # print(f'a: {a}')
        return flatted_packs[sentinel-1],sentinel
    return model,sentinel