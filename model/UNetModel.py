import numpy as np
import tensorflow as tf

from . import BaseModel

class UNetModel(BaseModel):
    """Implement the U-Net model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._setup()

    def _set_model(self):
        x1 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(self.inputs)    #256
        x1 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(x1)             #256
        p1 = tf.keras.layers.MaxPool2D(2)(x1)                                                           #128

        x2 = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(p1)            #128
        x2 = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(x2)            #128
        p2 = tf.keras.layers.MaxPool2D(2)(x2)                                                           #64

        x3 = tf.keras.layers.Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')(p2)            #64
        x3 = tf.keras.layers.Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')(x3)            #64
        p3 = tf.keras.layers.MaxPool2D(2)(x3)                                                           #32

        x4 = tf.keras.layers.Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')(p3)            #32
        x4 = tf.keras.layers.Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')(x4)            #32
        p4 = tf.keras.layers.MaxPool2D(2)(x4)                                                           #16

        x5 = tf.keras.layers.Conv2D(1024, 3, activation=tf.nn.leaky_relu, padding='same')(p4)           #16
        x5 = tf.keras.layers.Conv2D(1024, 3, activation=tf.nn.leaky_relu, padding='same')(x5)           #16
        d5 = tf.keras.layers.Dropout(0.5)(x5)                                                           #16

        up4 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(d5) #32
        concat4 = tf.concat([x4, up4], axis=3)                                                          #32
        c4 = tf.keras.layers.Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')(concat4)       #32
        c4 = tf.keras.layers.Conv2D(512, 3, activation=tf.nn.leaky_relu, padding='same')(c4)            #32

        up3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(c4) #64
        concat3 = tf.concat([x3, up3], axis=3)                                                          #64
        c3 = tf.keras.layers.Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')(concat3)       #64
        c3 = tf.keras.layers.Conv2D(256, 3, activation=tf.nn.leaky_relu, padding='same')(c3)            #64

        up2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(c3) #128
        concat2 = tf.concat([x2, up2], axis=3)                                                          #128
        c2 = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(concat2)       #128
        c2 = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(c2)            #128

        up1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, padding='same')(c2) #256
        concat1 = tf.concat([x1, up1], axis=3)                                                          #256      
        c1 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(concat1)        #256
        c1 = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(c1)             #256

        x = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.leaky_relu, padding='same')(c1)               #256
        self.outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax, padding='same')(x)        #256