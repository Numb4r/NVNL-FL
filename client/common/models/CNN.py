from tensorflow.keras import layers,models
import tensorflow as tf

def get_model():
    model = tf.keras.models.Sequential()
    
   # Convolutional Block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape[1:]))
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
#     model = tf.keras.models.Sequential([
#            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#            layers.MaxPooling2D((2, 2)),
#            layers.Conv2D(64, (3, 3), activation='relu'),
#            layers.MaxPooling2D((2, 2)),
#            layers.Conv2D(128, (3, 3), activation='relu'),
#            layers.MaxPooling2D((2, 2)),
#            layers.Flatten(),
#            layers.Dense(128, activation='relu'),
#            layers.Dense(10, activation='softmax') 
#         ])
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model