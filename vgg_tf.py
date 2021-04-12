from tensorflow.keras.layers import Activation, Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Sequential

cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}
def VGG():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False))
    model.add(Activation('relu'))
    in_channels=64
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu')) 
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu')) 
    model.add(Conv2D(256, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu')) 
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", strides=(1, 1), use_bias=False))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512))   
    model.add(Activation('relu'))
    model.add(Dense(512))   
    model.add(Activation('relu'))
    model.add(Dense(10))   
    return model